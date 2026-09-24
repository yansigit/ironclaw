//! Cursor AgentService Run client (`LlmProvider`).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use futures::{StreamExt, stream};
use reqwest::header::{
    ACCEPT, AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue, USER_AGENT,
};
use reqwest::{Body, Client};
use rust_decimal::Decimal;
use secrecy::ExposeSecret as _;
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::config::{CursorConfig, hardened_streaming_client_builder};
use crate::cursor_wire::{
    AgentRunInput, ExecEvent, KvEvent, ServerEvent, cursor_agent_origin_from_server_config,
    decode_connect_frames, decode_server_payload, encode_agent_run, encode_exec_rejected,
    encode_kv_get_result, encode_kv_set_result, encode_request_context_response,
    encode_shell_rejected, visible_composer_text,
};
use crate::error::LlmError;
use crate::provider::{
    ChatMessage, CompletionRequest, CompletionResponse, FinishReason, LlmProvider, Role,
    ToolCompletionRequest, ToolCompletionResponse,
};

const RUN_PATH: &str = "/agent.v1.AgentService/Run";
const SERVER_CONFIG_PATH: &str = "/aiserver.v1.ServerConfigService/GetServerConfig";
const CLIENT_VERSION: &str = "cli-2026.07.08-0c04a8a";

pub struct CursorProvider {
    config: CursorConfig,
    client: Client,
    stream_idle_timeout: Duration,
    agent_origin: tokio::sync::OnceCell<String>,
}

impl CursorProvider {
    pub fn new(config: CursorConfig, request_timeout_secs: u64) -> Result<Self, LlmError> {
        let client = hardened_streaming_client_builder()
            .build()
            .map_err(|error| LlmError::RequestFailed {
                provider: "cursor".to_string(),
                reason: format!("Failed to create HTTP client: {error}"),
            })?;
        Ok(Self {
            config,
            client,
            stream_idle_timeout: Duration::from_secs(request_timeout_secs),
            agent_origin: tokio::sync::OnceCell::new(),
        })
    }

    fn server_config_headers(access_token: &str) -> Result<HeaderMap, LlmError> {
        let mut headers = HeaderMap::new();
        headers.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/proto"),
        );
        headers.insert(
            HeaderName::from_static("connect-protocol-version"),
            HeaderValue::from_static("1"),
        );
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {access_token}")).map_err(|error| {
                LlmError::RequestFailed {
                    provider: "cursor".to_string(),
                    reason: format!("Invalid authorization header: {error}"),
                }
            })?,
        );
        headers.insert(
            HeaderName::from_static("x-cursor-client-type"),
            HeaderValue::from_static("cli"),
        );
        headers.insert(
            HeaderName::from_static("x-cursor-client-version"),
            HeaderValue::from_static(CLIENT_VERSION),
        );
        Ok(headers)
    }

    async fn resolve_agent_origin(&self, access_token: &str) -> Result<String, LlmError> {
        self.agent_origin
            .get_or_try_init(|| async {
                let url = format!(
                    "{}{}",
                    self.config.base_url.trim_end_matches('/'),
                    SERVER_CONFIG_PATH
                );
                let response = self
                    .client
                    .post(url)
                    .headers(Self::server_config_headers(access_token)?)
                    .body(Vec::new())
                    .send()
                    .await
                    .map_err(|error| {
                        LlmError::RequestFailed {
                            provider: "cursor".to_string(),
                            reason: format!("Cursor GetServerConfig failed: {error}"),
                        }
                    })?;

                if response.status() != reqwest::StatusCode::OK {
                    let status = response.status();
                    let body = response
                        .bytes()
                        .await
                        .map(|b| b.to_vec())
                        .unwrap_or_default();
                    let mut reason = format!("Cursor GetServerConfig HTTP {status}");
                    let snippet = sanitize_error_body(&body, access_token);
                    if !snippet.is_empty() {
                        reason.push_str(": ");
                        reason.push_str(&snippet);
                    }
                    return Err(LlmError::RequestFailed {
                        provider: "cursor".to_string(),
                        reason,
                    });
                }

                let body = response.bytes().await.map_err(|error| {
                    LlmError::RequestFailed {
                        provider: "cursor".to_string(),
                        reason: format!("Cursor GetServerConfig body: {error}"),
                    }
                })?;

                cursor_agent_origin_from_server_config(&body).map_err(|error| {
                    LlmError::RequestFailed {
                        provider: "cursor".to_string(),
                        reason: format!("Cursor GetServerConfig parse error: {error}"),
                    }
                })
            })
            .await
            .map(|origin| origin.clone())
    }

    fn model_for_request(&self, override_model: Option<&str>) -> String {
        override_model
            .map(str::trim)
            .filter(|m| !m.is_empty() && !m.eq_ignore_ascii_case("default"))
            .map(str::to_string)
            .unwrap_or_else(|| self.config.model.clone())
    }

    async fn access_token(&self) -> Result<String, LlmError> {
        if let Some(token) = self
            .config
            .access_token
            .as_ref()
            .map(|t| t.expose_secret().trim())
            .filter(|t| !t.is_empty())
        {
            return Ok(token.to_string());
        }
        if let Ok(token) = std::env::var("CURSOR_ACCESS_TOKEN") {
            let token = token.trim();
            if !token.is_empty() {
                return Ok(token.to_string());
            }
        }
        if let Some(path) = &self.config.session_path {
            let data = tokio::fs::read_to_string(path).await.map_err(|_error| {
                LlmError::AuthFailed {
                    provider: "cursor".to_string(),
                }
            })?;
            return access_token_from_session_json(&data);
        }
        Err(LlmError::AuthFailed {
            provider: "cursor".to_string(),
        })
    }

    fn run_headers(&self, access_token: &str) -> Result<HeaderMap, LlmError> {
        let mut headers = HeaderMap::new();
        headers.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/connect+proto"),
        );
        headers.insert(
            HeaderName::from_static("connect-protocol-version"),
            HeaderValue::from_static("1"),
        );
        headers.insert(HeaderName::from_static("te"), HeaderValue::from_static("trailers"));
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {access_token}")).map_err(|error| {
                LlmError::RequestFailed {
                    provider: "cursor".to_string(),
                    reason: format!("Invalid authorization header: {error}"),
                }
            })?,
        );
        headers.insert(
            HeaderName::from_static("x-ghost-mode"),
            HeaderValue::from_static("true"),
        );
        headers.insert(
            HeaderName::from_static("x-cursor-client-type"),
            HeaderValue::from_static("cli"),
        );
        headers.insert(
            HeaderName::from_static("x-cursor-client-version"),
            HeaderValue::from_static(CLIENT_VERSION),
        );
        headers.insert(
            HeaderName::from_static("x-request-id"),
            HeaderValue::from_str(&Uuid::new_v4().to_string()).map_err(|error| {
                LlmError::RequestFailed {
                    provider: "cursor".to_string(),
                    reason: format!("Invalid x-request-id: {error}"),
                }
            })?,
        );
        headers.insert(
            HeaderName::from_static("x-session-id"),
            HeaderValue::from_str(&Uuid::new_v4().to_string()).map_err(|error| {
                LlmError::RequestFailed {
                    provider: "cursor".to_string(),
                    reason: format!("Invalid x-session-id: {error}"),
                }
            })?,
        );
        headers.insert(ACCEPT, HeaderValue::from_static("application/connect+proto"));
        headers.insert(USER_AGENT, HeaderValue::from_static("ironclaw/cursor-provider"));
        Ok(headers)
    }

    fn flatten_messages(messages: &[ChatMessage]) -> (Option<String>, String) {
        let mut system_parts = Vec::new();
        let mut user_parts = Vec::new();
        for message in messages {
            match message.role {
                Role::System => {
                    if !message.content.is_empty() {
                        system_parts.push(message.content.clone());
                    }
                }
                Role::User | Role::HostReminder => {
                    if message.role == Role::User && !message.content.is_empty() {
                        user_parts.push(format!("User: {}", message.content));
                    } else if !message.content.is_empty() {
                        user_parts.push(message.content.clone());
                    }
                }
                Role::Assistant => {
                    if !message.content.is_empty() {
                        user_parts.push(format!("Assistant: {}", message.content));
                    }
                }
                Role::Tool => {
                    let name = message.name.as_deref().unwrap_or("tool");
                    let id = message.tool_call_id.as_deref().unwrap_or("");
                    user_parts.push(format!(
                        "<tool_result>\n<tool_name>{name}</tool_name>\n<tool_call_id>{id}</tool_call_id>\n<result>{content}</result>\n</tool_result>",
                        name = name,
                        id = id,
                        content = message.content
                    ));
                }
            }
        }
        let system_prompt = if system_parts.is_empty() {
            None
        } else {
            Some(system_parts.join("\n"))
        };
        (system_prompt, user_parts.join("\n"))
    }

    async fn run_turn(
        &self,
        model_id: &str,
        messages: &[ChatMessage],
        metadata: &HashMap<String, String>,
    ) -> Result<String, LlmError> {
        let (system_prompt, user_text) = Self::flatten_messages(messages);
        let conversation_id = metadata
            .get("session_id")
            .filter(|v| !v.trim().is_empty())
            .cloned()
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        let message_id = Uuid::new_v4().to_string();
        let encoded = encode_agent_run(&AgentRunInput {
            model_id: model_id.to_string(),
            user_text,
            conversation_id,
            message_id,
            system_prompt,
        });
        let access_token = self.access_token().await?;
        let agent_origin = self.resolve_agent_origin(&access_token).await?;
        let url = format!("{}{}", agent_origin.trim_end_matches('/'), RUN_PATH);
        let headers = self.run_headers(&access_token)?;

        let (body_tx, body_rx) = mpsc::unbounded_channel::<Vec<u8>>();
        body_tx
            .send(encoded.frame.clone())
            .map_err(|_| request_failed("Cursor request body closed before first frame"))?;

        let request_body = Body::wrap_stream(stream::unfold(body_rx, |mut rx| async move {
            match rx.recv().await {
                Some(chunk) => Some((Ok::<Bytes, std::io::Error>(Bytes::from(chunk)), rx)),
                None => None,
            }
        }));

        let response = self
            .client
            .post(url)
            .headers(headers)
            .body(request_body)
            .send()
            .await
            .map_err(|error| {
                LlmError::RequestFailed {
                    provider: "cursor".to_string(),
                    reason: error.to_string(),
                }
            })?;

        if response.status() != reqwest::StatusCode::OK {
            let status = response.status();
            let body = response
                .bytes()
                .await
                .map(|b| b.to_vec())
                .unwrap_or_default();
            let mut reason = format!("HTTP {status}");
            let snippet = sanitize_error_body(&body, &access_token);
            if !snippet.is_empty() {
                reason.push_str(": ");
                reason.push_str(&snippet);
            }
            return Err(LlmError::RequestFailed {
                provider: "cursor".to_string(),
                reason,
            });
        }

        let content_type = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string)
            .unwrap_or_default();

        let mut frame_buf = Vec::new();
        let mut raw_body = Vec::new();
        let mut text_buf = String::new();
        let mut thinking_buf = String::new();
        let mut saw_content = false;
        let mut turn_ended = false;
        let blobs = encoded.blobs;

        let mut response_stream = response.bytes_stream();
        while !turn_ended {
            let next_chunk = tokio::time::timeout(self.stream_idle_timeout, response_stream.next())
                .await
                .map_err(|_| {
                    LlmError::RequestFailed {
                        provider: "cursor".to_string(),
                        reason: "Cursor response idle timeout".to_string(),
                    }
                })?;
            match next_chunk {
                Some(Ok(chunk)) => {
                    raw_body.extend_from_slice(&chunk);
                    frame_buf.extend_from_slice(&chunk);
                    drain_connect_frames(&mut frame_buf, |payload| {
                        process_server_payload(
                            payload,
                            &blobs,
                            &body_tx,
                            &mut text_buf,
                            &mut thinking_buf,
                            &mut saw_content,
                            &mut turn_ended,
                        )
                    })?;
                }
                Some(Err(error)) => {
                    return Err(LlmError::RequestFailed {
                        provider: "cursor".to_string(),
                        reason: error.to_string(),
                    });
                }
                None => break,
            }
        }

        if !frame_buf.is_empty() {
            drain_connect_frames(&mut frame_buf, |payload| {
                process_server_payload(
                    payload,
                    &blobs,
                    &body_tx,
                    &mut text_buf,
                    &mut thinking_buf,
                    &mut saw_content,
                    &mut turn_ended,
                )
            })?;
        }

        let visible = visible_text(model_id, &text_buf, &thinking_buf);
        if visible.trim().is_empty() {
            return Err(LlmError::RequestFailed {
                provider: "cursor".to_string(),
                reason: empty_completion_reason(&raw_body, &content_type, &access_token),
            });
        }
        Ok(visible)
    }
}

fn request_failed(reason: impl Into<String>) -> LlmError {
    LlmError::RequestFailed {
        provider: "cursor".to_string(),
        reason: reason.into(),
    }
}

fn empty_completion_reason(body: &[u8], content_type: &str, token: &str) -> String {
    let len = body.len();
    if len == 0 {
        return format!("Cursor returned an empty completion (0 bytes)");
    }
    if decode_connect_frames(body).is_err() {
        let snippet = sanitize_error_body(body, token);
        let snippet = if snippet.len() > 200 {
            snippet[..200].to_string()
        } else {
            snippet
        };
        return format!(
            "Cursor returned an empty completion ({len} bytes); content-type: {content_type}; body: {snippet}"
        );
    }
    format!("Cursor returned an empty completion ({len} bytes)")
}

fn sanitize_error_body(body: &[u8], token: &str) -> String {
    let text = String::from_utf8_lossy(body);
    let redacted = if token.is_empty() {
        text.to_string()
    } else {
        text.replace(token, "[REDACTED]")
    };
    if redacted.len() > 500 {
        redacted[..500].to_string()
    } else {
        redacted
    }
}

fn visible_text(model_id: &str, text_buf: &str, thinking_buf: &str) -> String {
    if model_id.starts_with("composer") {
        let from_thinking = visible_composer_text(thinking_buf);
        if !from_thinking.is_empty() {
            return from_thinking;
        }
    }
    text_buf.to_string()
}

fn drain_connect_frames(
    buffer: &mut Vec<u8>,
    mut on_payload: impl FnMut(&[u8]) -> Result<(), LlmError>,
) -> Result<(), LlmError> {
    while !buffer.is_empty() {
        if buffer.len() < 5 {
            return Ok(());
        }
        let len = u32::from_be_bytes([buffer[1], buffer[2], buffer[3], buffer[4]]) as usize;
        if buffer.len() < 5 + len {
            return Ok(());
        }
        let frame = buffer.drain(..5 + len).collect::<Vec<_>>();
        let payloads = decode_connect_frames(&frame).map_err(|_| {
            LlmError::RequestFailed {
                provider: "cursor".to_string(),
                reason: "Invalid Cursor connect frame".to_string(),
            }
        })?;
        for payload in payloads {
            on_payload(&payload)?;
        }
    }
    Ok(())
}

fn process_server_payload(
    payload: &[u8],
    blobs: &HashMap<String, Vec<u8>>,
    body_tx: &mpsc::UnboundedSender<Vec<u8>>,
    text_buf: &mut String,
    thinking_buf: &mut String,
    saw_content: &mut bool,
    turn_ended: &mut bool,
) -> Result<(), LlmError> {
    for event in decode_server_payload(payload) {
        match event {
            ServerEvent::Text { text } => {
                text_buf.push_str(&text);
                *saw_content = true;
            }
            ServerEvent::Thinking { text } => {
                thinking_buf.push_str(&text);
                *saw_content = true;
            }
            ServerEvent::TurnEnded => {
                *turn_ended = true;
            }
            ServerEvent::Heartbeat | ServerEvent::Ignore => {}
            ServerEvent::Exec(exec) => {
                for reply in reply_for_exec(&exec) {
                    body_tx.send(reply).map_err(|_| {
                        request_failed("Cursor request body closed while sending exec reply")
                    })?;
                }
            }
            ServerEvent::Kv(kv) => {
                for reply in reply_for_kv(&kv, blobs) {
                    body_tx.send(reply).map_err(|_| {
                        request_failed("Cursor request body closed while sending kv reply")
                    })?;
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn reply_for_server_payload(
    payload: &[u8],
    blobs: &HashMap<String, Vec<u8>>,
) -> Vec<Vec<u8>> {
    let server_payload = decode_connect_frames(payload)
        .ok()
        .and_then(|frames| frames.first().cloned())
        .unwrap_or_else(|| payload.to_vec());
    let mut out = Vec::new();
    for event in decode_server_payload(&server_payload) {
        match event {
            ServerEvent::Exec(exec) => out.extend(reply_for_exec(&exec)),
            ServerEvent::Kv(kv) => out.extend(reply_for_kv(&kv, blobs)),
            _ => {}
        }
    }
    out
}

fn reply_for_exec(exec: &ExecEvent) -> Vec<Vec<u8>> {
    match exec {
        ExecEvent::RequestContext { exec_msg_id, exec_id } => {
            vec![encode_request_context_response(*exec_msg_id, exec_id)]
        }
        ExecEvent::Shell {
            exec_msg_id,
            exec_id,
            command,
            working_dir,
        } => vec![encode_shell_rejected(*exec_msg_id, exec_id, command, working_dir)],
        ExecEvent::Other {
            exec_msg_id,
            exec_id,
            variant_field,
        } => vec![encode_exec_rejected(*exec_msg_id, exec_id, *variant_field)],
    }
}

fn reply_for_kv(kv: &KvEvent, blobs: &HashMap<String, Vec<u8>>) -> Vec<Vec<u8>> {
    match kv {
        KvEvent::GetBlob {
            kv_id,
            blob_id,
            request_metadata,
        } => {
            let key = hex::encode(blob_id);
            let data = blobs
                .get(&key)
                .map(|v| v.as_slice())
                .unwrap_or(&[] as &[u8]);
            vec![encode_kv_get_result(
                *kv_id,
                data,
                request_metadata.as_deref(),
            )]
        }
        KvEvent::SetBlob {
            kv_id,
            request_metadata,
        } => vec![encode_kv_set_result(*kv_id, request_metadata.as_deref())],
    }
}

fn access_token_from_session_json(data: &str) -> Result<String, LlmError> {
    let value: serde_json::Value = serde_json::from_str(data).map_err(|_| LlmError::AuthFailed {
        provider: "cursor".to_string(),
    })?;
    for key in ["accessToken", "access_token"] {
        if let Some(token) = value.get(key).and_then(|v| v.as_str()) {
            let token = token.trim();
            if !token.is_empty() {
                return Ok(token.to_string());
            }
        }
    }
    Err(LlmError::AuthFailed {
        provider: "cursor".to_string(),
    })
}

pub fn create_cursor_provider(config: &crate::config::LlmConfig) -> Result<Arc<dyn LlmProvider>, LlmError> {
    let cursor = config
        .cursor
        .clone()
        .ok_or_else(|| LlmError::AuthFailed {
            provider: "cursor".to_string(),
        })?;
    Ok(Arc::new(
        CursorProvider::new(cursor, config.request_timeout_secs)?,
    ))
}

#[async_trait]
impl LlmProvider for CursorProvider {
    fn provider_id(&self) -> String {
        "cursor".to_string()
    }

    fn model_name(&self) -> &str {
        &self.config.model
    }

    fn cost_per_token(&self) -> (Decimal, Decimal) {
        (Decimal::ZERO, Decimal::ZERO)
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let model = request
            .model
            .as_deref()
            .map(|m| self.model_for_request(Some(m)))
            .unwrap_or_else(|| self.model_for_request(None));
        let content = self
            .run_turn(&model, &request.messages, &request.metadata)
            .await?;
        Ok(CompletionResponse {
            content,
            input_tokens: 0,
            output_tokens: 0,
            finish_reason: FinishReason::Stop,
            reasoning: None,
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 0,
        })
    }

    async fn complete_with_tools(
        &self,
        request: ToolCompletionRequest,
    ) -> Result<ToolCompletionResponse, LlmError> {
        let model = request
            .model
            .as_deref()
            .map(|m| self.model_for_request(Some(m)))
            .unwrap_or_else(|| self.model_for_request(None));
        let content = self
            .run_turn(&model, &request.messages, &request.metadata)
            .await?;
        Ok(ToolCompletionResponse {
            content: Some(content),
            tool_calls: Vec::new(),
            input_tokens: 0,
            output_tokens: 0,
            finish_reason: FinishReason::Stop,
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 0,
            reasoning: None,
            reasoning_details: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cursor_wire::protobuf_string_path;
    use crate::provider::{ChatMessage, CompletionRequest};

    #[test]
    fn shell_server_frame_is_rejected() {
        let frame = crate::cursor_wire::encode_shell_server_frame(4, "e1", "uname", "/tmp");
        let replies = reply_for_server_payload(&frame, &std::collections::HashMap::new());
        assert_eq!(replies.len(), 1);
        let frames = crate::cursor_wire::decode_connect_frames(&replies[0]).expect("reply frame");
        let reason = protobuf_string_path(&frames[0], &[2, 2, 2, 3]);
        assert_eq!(
            reason.as_deref(),
            Some(crate::cursor_wire::LOCAL_TOOL_REJECTION)
        );
        let source = include_str!("cursor.rs");
        let forbidden_shell = ["Command", "::", "new"].concat();
        let forbidden_process = ["std", "::process"].concat();
        assert!(!source.contains(&forbidden_shell));
        assert!(!source.contains(&forbidden_process));
    }

    fn live_provider() -> CursorProvider {
        let token = std::env::var("CURSOR_ACCESS_TOKEN")
            .expect("CURSOR_ACCESS_TOKEN is required");
        let config = CursorConfig::build(
            Some(crate::config::CursorConfig::DEFAULT_MODEL.to_string()),
            None,
            Some(secrecy::SecretString::from(token)),
            None,
        );
        CursorProvider::new(config, crate::config::DEFAULT_REQUEST_TIMEOUT_SECS).expect("provider")
    }

    fn user_request(text: &str) -> CompletionRequest {
        CompletionRequest::new(vec![ChatMessage::user(text)])
    }

    fn response_text(response: &CompletionResponse) -> &str {
        &response.content
    }

    #[tokio::test]
    #[ignore]
    async fn live_composer_smoke() {
        let provider = live_provider();
        let response = provider
            .complete(
                CompletionRequest::new(vec![ChatMessage::user(
                    "Reply with the single word pong.",
                )]).with_model("composer-2.5"),
            )
            .await
            .expect("smoke completion");
        let text = response_text(&response).to_ascii_lowercase();
        assert!(text.contains("pong"), "smoke text was {text}");
    }

    #[tokio::test]
    #[ignore]
    async fn live_composer_second_turn() {
        let provider = live_provider();
        let first = provider
            .complete(user_request("Reply with the single word alpha."))
            .await
            .expect("first");
        assert!(!response_text(&first).trim().is_empty());
        let second = provider
            .complete(user_request("Reply with the single word beta."))
            .await
            .expect("second");
        let text = response_text(&second).to_ascii_lowercase();
        assert!(text.contains("beta"), "second turn text was {text}");
    }
}
