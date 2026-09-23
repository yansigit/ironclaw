use std::time::Duration;

use async_trait::async_trait;
use rust_decimal::Decimal;
use secrecy::ExposeSecret as _;
use sha2::{Digest, Sha256};

use crate::config::OpenCodeGoConfig;
use crate::error::{LlmError, ProductionModelAdapter, ProviderHttpError, map_provider_http_error};
use crate::provider::{
    ChatMessage, CompletionRequest, CompletionResponse, CompletionResponseFormat, ContentPart,
    FinishReason, LlmProvider, ReasoningDetails, Role, ToolCall, ToolCompletionRequest,
    ToolCompletionResponse, ToolDefinition, map_provider_finish_token,
    openai_json_schema_response_format,
};

pub const SESSION_HEADER: &str = "x-opencode-session";

pub enum OpenCodeGoWire {
    ChatCompletions,
    Responses,
}

pub fn wire_for_model(model: &str) -> OpenCodeGoWire {
    match model {
        "gpt-5.6-luna"
        | "grok-4.6"
        | "muse-spark-1.3-contributor"
        | "muse-spark-1.2-contributor" => OpenCodeGoWire::Responses,
        _ => OpenCodeGoWire::ChatCompletions,
    }
}

pub fn session_header_value(lane: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"ironclaw/opencode-go/session/v1\0");
    hasher.update(lane.as_bytes());
    format!("ic_{}", hex::encode(&hasher.finalize()[..16]))
}

pub struct OpenCodeGoProvider {
    config: OpenCodeGoConfig,
    http: reqwest::Client,
}

impl OpenCodeGoProvider {
    pub fn new(config: OpenCodeGoConfig, request_timeout_secs: u64) -> Result<Self, LlmError> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(request_timeout_secs))
            .build()
            .map_err(|error| LlmError::RequestFailed {
                provider: "opencode_go".to_string(),
                reason: error.to_string(),
            })?;
        Ok(Self { config, http })
    }

    fn api_key(&self) -> Result<&str, LlmError> {
        match self
            .config
            .api_key
            .as_ref()
            .map(|key| key.expose_secret().trim())
            .filter(|key| !key.is_empty())
        {
            Some(key) => Ok(key),
            None => Err(LlmError::AuthFailed {
                provider: "opencode_go".to_string(),
            }),
        }
    }

    fn lane_from_metadata(&self, metadata: &std::collections::HashMap<String, String>) -> String {
        metadata
            .get("session_id")
            .map(|value| value.trim())
            .filter(|value| !value.is_empty())
            .unwrap_or("ironclaw")
            .to_string()
    }

    fn message_json(message: &ChatMessage) -> serde_json::Value {
        let role = match message.role {
            Role::System => "system",
            Role::User | Role::HostReminder => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        };
        let mut value = serde_json::json!({
            "role": role,
            "content": Self::content_json(message),
        });
        if let Some(tool_call_id) = &message.tool_call_id {
            value["tool_call_id"] = serde_json::Value::String(tool_call_id.clone());
        }
        if let Some(name) = &message.name {
            value["name"] = serde_json::Value::String(name.clone());
        }
        if let Some(calls) = &message.tool_calls {
            value["tool_calls"] = serde_json::Value::Array(
                calls
                    .iter()
                    .map(|call| {
                        serde_json::json!({
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.arguments.to_string(),
                            }
                        })
                    })
                    .collect(),
            );
        }
        if let Some(reasoning) = message
            .reasoning
            .as_ref()
            .filter(|text| !text.trim().is_empty())
        {
            value["reasoning_content"] = serde_json::Value::String(reasoning.clone());
        }
        value
    }

    fn content_json(message: &ChatMessage) -> serde_json::Value {
        if message.content_parts.is_empty() {
            if message.role == Role::Assistant && message.content.is_empty() {
                return serde_json::Value::Null;
            }
            return serde_json::Value::String(message.content.clone());
        }
        let mut parts = Vec::new();
        if !message.content.is_empty() {
            parts.push(serde_json::json!({"type": "text", "text": message.content}));
        }
        for part in &message.content_parts {
            match part {
                ContentPart::Text { text } => {
                    parts.push(serde_json::json!({"type": "text", "text": text}));
                }
                ContentPart::ImageUrl { image_url } => {
                    parts.push(serde_json::json!({
                        "type": "image_url",
                        "image_url": {
                            "url": image_url.url,
                            "detail": image_url.normalized_openai_detail(),
                        }
                    }));
                }
            }
        }
        serde_json::Value::Array(parts)
    }

    fn encode_tool_choice(choice: &str) -> serde_json::Value {
        match choice {
            "auto" | "required" | "none" => serde_json::Value::String(choice.to_string()),
            specific => serde_json::json!({
                "type": "function",
                "function": {"name": specific}
            }),
        }
    }

    fn body(
        model: &str,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        stop: Option<&[String]>,
        response_format: Option<&CompletionResponseFormat>,
        tool_choice: Option<&str>,
    ) -> Result<serde_json::Value, LlmError> {
        if tool_choice.is_some() && tools.is_empty() {
            return Err(LlmError::InvalidRequest {
                provider: "opencode_go".to_string(),
                reason: "tool_choice requires at least one tool".to_string(),
            });
        }
        let mut body = serde_json::json!({
            "model": model,
            "messages": messages.iter().map(Self::message_json).collect::<Vec<_>>(),
        });
        if let Some(max_tokens) = max_tokens {
            body["max_tokens"] = serde_json::json!(max_tokens);
        }
        if let Some(temperature) = temperature {
            body["temperature"] = serde_json::json!(temperature);
        }
        if let Some(stop) = stop.filter(|stop| !stop.is_empty()) {
            body["stop"] = serde_json::json!(stop);
        }
        if let Some(format) = response_format {
            body["response_format"] = openai_json_schema_response_format(format.clone());
        }
        if !tools.is_empty() {
            body["tools"] = serde_json::Value::Array(
                tools
                    .iter()
                    .map(|tool| {
                        serde_json::json!({
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.parameters,
                            }
                        })
                    })
                    .collect(),
            );
        }
        if let Some(choice) = tool_choice {
            body["tool_choice"] = Self::encode_tool_choice(choice);
        }
        Ok(body)
    }

    async fn post_chat(&self, call: &ChatCall<'_>) -> Result<serde_json::Value, LlmError> {
        let api_key = self.api_key()?;
        if matches!(wire_for_model(call.model), OpenCodeGoWire::Responses) {
            return Err(LlmError::InvalidRequest {
                provider: "opencode_go".to_string(),
                reason: format!(
                    "model {} uses the OpenCode Go responses wire, which this provider does not send",
                    call.model
                ),
            });
        }
        let body = Self::body(
            call.model,
            call.messages,
            call.tools,
            call.max_tokens,
            call.temperature,
            call.stop,
            call.response_format,
            call.tool_choice,
        )?;
        let url = format!(
            "{}/chat/completions",
            self.config.base_url.trim_end_matches('/')
        );
        let response = self
            .http
            .post(url)
            .bearer_auth(api_key)
            .header(SESSION_HEADER, session_header_value(call.lane))
            .json(&body)
            .send()
            .await
            .map_err(|error| LlmError::RequestFailed {
                provider: "opencode_go".to_string(),
                reason: error.to_string(),
            })?;
        let status = response.status();
        let retry_after = crate::retry::retry_after_for_status(
            status.as_u16(),
            response.headers().get(reqwest::header::RETRY_AFTER),
        );
        let text = response
            .text()
            .await
            .map_err(|error| LlmError::RequestFailed {
                provider: "opencode_go".to_string(),
                reason: error.to_string(),
            })?;
        if !status.is_success() {
            return Err(map_provider_http_error(ProviderHttpError {
                adapter: ProductionModelAdapter::OpenCodeGo,
                model: call.model,
                status: status.as_u16(),
                body: &text,
                retry_after,
            }));
        }
        serde_json::from_str(&text).map_err(|error| LlmError::InvalidResponse {
            provider: "opencode_go".to_string(),
            reason: format!(
                "JSON parse error: {error}. Raw: {}",
                ironclaw_common::truncate_for_preview(&text, 512)
            ),
        })
    }

    fn first_choice(payload: &serde_json::Value) -> Result<&serde_json::Value, LlmError> {
        payload["choices"]
            .as_array()
            .and_then(|choices| choices.first())
            .ok_or_else(|| LlmError::EmptyResponse {
                provider: "opencode_go".to_string(),
            })
    }

    fn parsed_tool_calls(choice: &serde_json::Value) -> Vec<ToolCall> {
        let mut tool_calls = Vec::new();
        if let Some(calls) = choice["message"]["tool_calls"].as_array() {
            for call in calls {
                let name = call["function"]["name"].as_str().unwrap_or("").to_string();
                let raw_args = call["function"]["arguments"].as_str().unwrap_or("{}");
                let (arguments, arguments_parse_error) =
                    match serde_json::from_str::<serde_json::Value>(raw_args) {
                        Ok(value) => (value, None),
                        Err(error) => (serde_json::json!({}), Some(error.to_string())),
                    };
                tool_calls.push(ToolCall {
                    id: call["id"].as_str().unwrap_or("").to_string(),
                    name,
                    arguments,
                    reasoning: None,
                    signature: None,
                    arguments_parse_error,
                });
            }
        }
        tool_calls
    }

    fn parsed_reasoning(choice: &serde_json::Value) -> Option<String> {
        choice["message"]["reasoning_content"]
            .as_str()
            .filter(|text| !text.trim().is_empty())
            .map(str::to_string)
    }

    fn usage_tokens(payload: &serde_json::Value) -> (u32, u32) {
        (
            payload["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32,
            payload["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32,
        )
    }

    async fn fetch_model_ids(&self) -> Result<Vec<String>, LlmError> {
        let api_key = self.api_key()?;
        let url = format!("{}/models", self.config.base_url.trim_end_matches('/'));
        let response = self
            .http
            .get(url)
            .bearer_auth(api_key)
            .header(SESSION_HEADER, session_header_value("ironclaw"))
            .send()
            .await
            .map_err(|error| LlmError::RequestFailed {
                provider: "opencode_go".to_string(),
                reason: error.to_string(),
            })?;
        let status = response.status();
        let retry_after = crate::retry::retry_after_for_status(
            status.as_u16(),
            response.headers().get(reqwest::header::RETRY_AFTER),
        );
        let text = response
            .text()
            .await
            .map_err(|error| LlmError::RequestFailed {
                provider: "opencode_go".to_string(),
                reason: error.to_string(),
            })?;
        if !status.is_success() {
            return Err(map_provider_http_error(ProviderHttpError {
                adapter: ProductionModelAdapter::OpenCodeGo,
                model: &self.config.model,
                status: status.as_u16(),
                body: &text,
                retry_after,
            }));
        }
        let payload: serde_json::Value =
            serde_json::from_str(&text).map_err(|error| LlmError::InvalidResponse {
                provider: "opencode_go".to_string(),
                reason: format!(
                    "JSON parse error: {error}. Raw: {}",
                    ironclaw_common::truncate_for_preview(&text, 512)
                ),
            })?;
        let Some(data) = payload["data"].as_array() else {
            return Err(LlmError::InvalidResponse {
                provider: "opencode_go".to_string(),
                reason: "response missing data array".to_string(),
            });
        };
        let mut ids = Vec::new();
        for entry in data {
            if let Some(id) = entry["id"].as_str().filter(|id| !id.is_empty()) {
                ids.push(id.to_string());
            }
        }
        Ok(ids)
    }
}

struct ChatCall<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    tools: &'a [ToolDefinition],
    lane: &'a str,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    stop: Option<&'a [String]>,
    response_format: Option<&'a CompletionResponseFormat>,
    tool_choice: Option<&'a str>,
}

#[async_trait]
impl LlmProvider for OpenCodeGoProvider {
    fn provider_id(&self) -> String {
        "opencode_go".to_string()
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
            .clone()
            .unwrap_or_else(|| self.config.model.clone());
        let lane = self.lane_from_metadata(&request.metadata);
        let stop = request.stop_sequences.as_deref();
        let call = ChatCall {
            model: &model,
            messages: &request.messages,
            tools: &[],
            lane: &lane,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            stop,
            response_format: request.response_format.as_ref(),
            tool_choice: None,
        };
        let payload = self.post_chat(&call).await?;
        let choice = Self::first_choice(&payload)?;
        let content = choice["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();
        let finish_reason = choice["finish_reason"]
            .as_str()
            .and_then(map_provider_finish_token)
            .unwrap_or(FinishReason::Unknown);
        let reasoning = Self::parsed_reasoning(choice);
        let (input_tokens, output_tokens) = Self::usage_tokens(&payload);
        Ok(CompletionResponse {
            content,
            input_tokens,
            output_tokens,
            finish_reason,
            reasoning,
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
            .clone()
            .unwrap_or_else(|| self.config.model.clone());
        let lane = self.lane_from_metadata(&request.metadata);
        let stop = request.stop_sequences.as_deref();
        let tool_choice = request.tool_choice.as_deref();
        let call = ChatCall {
            model: &model,
            messages: &request.messages,
            tools: &request.tools,
            lane: &lane,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            stop,
            response_format: request.response_format.as_ref(),
            tool_choice,
        };
        let payload = self.post_chat(&call).await?;
        let choice = Self::first_choice(&payload)?;
        let content = choice["message"]["content"].as_str().map(str::to_string);
        let tool_calls = Self::parsed_tool_calls(choice);
        let finish_reason = choice["finish_reason"]
            .as_str()
            .and_then(map_provider_finish_token)
            .unwrap_or(FinishReason::Unknown);
        let reasoning = Self::parsed_reasoning(choice);
        let reasoning_details = reasoning
            .as_ref()
            .and_then(|text| ReasoningDetails::from_text(text.clone()));
        let (input_tokens, output_tokens) = Self::usage_tokens(&payload);
        Ok(ToolCompletionResponse {
            content,
            tool_calls,
            input_tokens,
            output_tokens,
            finish_reason,
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 0,
            reasoning,
            reasoning_details,
        })
    }

    async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        self.fetch_model_ids().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use secrecy::SecretString;
    use std::collections::HashMap;

    fn loopback_config(addr: std::net::SocketAddr) -> OpenCodeGoConfig {
        OpenCodeGoConfig {
            model: "kimi-k2.7-code".to_string(),
            base_url: format!("http://{addr}"),
            api_key: Some(SecretString::from("test-go-key")),
        }
    }

    #[test]
    fn debug_redacts_the_api_key() {
        let debug = format!(
            "{:?}",
            OpenCodeGoConfig::build(None, None, Some(SecretString::from("sk-live-secret")))
        );
        assert!(!debug.contains("sk-live-secret"));
    }

    #[test]
    fn responses_models_are_not_sent_to_chat_completions() {
        for model in [
            "gpt-5.6-luna",
            "grok-4.6",
            "muse-spark-1.3-contributor",
            "muse-spark-1.2-contributor",
        ] {
            assert!(
                matches!(wire_for_model(model), OpenCodeGoWire::Responses),
                "{model}"
            );
        }
    }

    #[test]
    fn chat_models_stay_on_chat_completions() {
        for model in ["kimi-k2.7-code", "glm-5.2", "deepseek-v4-flash"] {
            assert!(
                matches!(wire_for_model(model), OpenCodeGoWire::ChatCompletions),
                "{model}"
            );
        }
    }

    #[test]
    fn session_header_hides_the_lane_and_is_stable() {
        let lane = "thread-secret-42";
        let header = session_header_value(lane);
        assert!(header.starts_with("ic_"));
        assert_eq!(header.len(), 3 + 32);
        assert!(!header.contains(lane));
        assert_eq!(header, session_header_value(lane));
        assert_ne!(header, session_header_value("thread-secret-43"));
        let mut hasher = Sha256::new();
        hasher.update(b"ironclaw/opencode-go/session/v1\0");
        hasher.update(lane.as_bytes());
        let expected = format!("ic_{}", hex::encode(&hasher.finalize()[..16]));
        assert_eq!(header, expected);
    }

    #[tokio::test]
    async fn complete_posts_chat_completions_with_session_header() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let mut buf = vec![0u8; 8192];
            let n = socket.read(&mut buf).await.unwrap();
            let request = String::from_utf8_lossy(&buf[..n]).to_string();
            assert!(request.starts_with("POST /chat/completions HTTP/1.1"));
            assert!(
                request.contains("authorization: Bearer test-go-key")
                    || request.contains("Authorization: Bearer test-go-key")
            );
            assert!(request.contains("x-opencode-session: "));
            assert!(request.contains("\"model\":\"kimi-k2.7-code\""));
            let body = "{\"choices\":[{\"message\":{\"content\":\"pong\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1}}";
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            socket.write_all(response.as_bytes()).await.unwrap();
            request
        });

        let provider = OpenCodeGoProvider::new(loopback_config(addr), 5).unwrap();
        let response = provider
            .complete(crate::provider::CompletionRequest::new(vec![
                crate::provider::ChatMessage::user("ping"),
            ]))
            .await
            .unwrap();
        assert_eq!(response.content, "pong");
        assert_eq!(response.input_tokens, 3);
        assert_eq!(response.output_tokens, 1);
        assert_eq!(response.finish_reason, crate::provider::FinishReason::Stop);
        let raw = server.await.unwrap();
        assert!(raw.contains("x-opencode-session: ic_"));
    }

    #[tokio::test]
    async fn list_models_gets_openai_data_with_session_header() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let mut buf = vec![0u8; 8192];
            let n = socket.read(&mut buf).await.unwrap();
            let request = String::from_utf8_lossy(&buf[..n]).to_string();
            assert!(request.starts_with("GET /models HTTP/1.1"));
            assert!(
                request.contains("authorization: Bearer test-go-key")
                    || request.contains("Authorization: Bearer test-go-key")
            );
            assert!(request.contains("x-opencode-session: ic_"));
            let body = r#"{"data":[{"id":"kimi-k2.7-code"},{"id":""},{"id":"gpt-5.6-luna"}]}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            socket.write_all(response.as_bytes()).await.unwrap();
            request
        });

        let provider = OpenCodeGoProvider::new(loopback_config(addr), 5).unwrap();
        let models = provider.list_models().await.unwrap();
        assert_eq!(
            models,
            vec!["kimi-k2.7-code".to_string(), "gpt-5.6-luna".to_string()]
        );
        let _raw = server.await.unwrap();
    }

    #[tokio::test]
    #[ignore = "calls OpenCode Go; requires OPENCODE_API_KEY"]
    async fn live_kimi_smoke() {
        let key = std::env::var("OPENCODE_API_KEY").expect("OPENCODE_API_KEY");
        let provider = OpenCodeGoProvider::new(
            crate::config::OpenCodeGoConfig::build(None, None, Some(SecretString::from(key))),
            60,
        )
        .unwrap();
        let response = provider
            .complete(crate::provider::CompletionRequest::new(vec![
                crate::provider::ChatMessage::user("Reply with the single word pong"),
            ]))
            .await
            .unwrap();
        assert!(!response.content.trim().is_empty());
    }

    #[tokio::test]
    async fn responses_model_is_invalid_request_without_http() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let accept = tokio::spawn(async move {
            tokio::time::timeout(Duration::from_millis(250), listener.accept()).await
        });

        let provider = OpenCodeGoProvider::new(loopback_config(addr), 5).unwrap();
        let err = provider
            .complete(
                crate::provider::CompletionRequest::new(vec![crate::provider::ChatMessage::user(
                    "ping",
                )])
                .with_model("gpt-5.6-luna"),
            )
            .await
            .unwrap_err();
        assert!(matches!(err, LlmError::InvalidRequest { .. }));
        assert!(
            accept.await.unwrap().is_err(),
            "responses-wire model must fail before opening a connection"
        );
    }

    #[tokio::test]
    async fn empty_api_key_is_auth_failed() {
        let provider = OpenCodeGoProvider::new(
            OpenCodeGoConfig {
                model: "kimi-k2.7-code".to_string(),
                base_url: "http://127.0.0.1:9".to_string(),
                api_key: None,
            },
            5,
        )
        .unwrap();
        let err = provider
            .complete(crate::provider::CompletionRequest::new(vec![
                crate::provider::ChatMessage::user("ping"),
            ]))
            .await
            .unwrap_err();
        assert!(matches!(err, LlmError::AuthFailed { .. }));
    }

    #[tokio::test]
    async fn http_401_is_auth_failed() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            use tokio::io::AsyncWriteExt;
            let body = "unauthorized";
            let response = format!(
                "HTTP/1.1 401 Unauthorized\r\ncontent-type: text/plain\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            socket.write_all(response.as_bytes()).await.unwrap();
        });

        let provider = OpenCodeGoProvider::new(loopback_config(addr), 5).unwrap();
        let err = provider
            .complete(crate::provider::CompletionRequest::new(vec![
                crate::provider::ChatMessage::user("ping"),
            ]))
            .await
            .unwrap_err();
        assert!(matches!(err, LlmError::AuthFailed { .. }));
    }

    #[tokio::test]
    async fn missing_choices_is_empty_response() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            use tokio::io::AsyncWriteExt;
            let body = r#"{"choices":[]}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            socket.write_all(response.as_bytes()).await.unwrap();
        });

        let provider = OpenCodeGoProvider::new(loopback_config(addr), 5).unwrap();
        let err = provider
            .complete(crate::provider::CompletionRequest::new(vec![
                crate::provider::ChatMessage::user("ping"),
            ]))
            .await
            .unwrap_err();
        assert!(matches!(err, LlmError::EmptyResponse { .. }));
    }

    #[tokio::test]
    async fn tool_history_posts_openai_fields() {
        use crate::provider::{
            CompletionResponseFormat, ContentPart, ImageUrl, ToolCompletionRequest, ToolDefinition,
        };
        use tokio::sync::oneshot;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = oneshot::channel();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let mut buf = vec![0u8; 65536];
            let n = socket.read(&mut buf).await.unwrap();
            let request = String::from_utf8_lossy(&buf[..n]).to_string();
            let body_start = request.find("\r\n\r\n").map(|idx| idx + 4).unwrap_or(0);
            let body = request[body_start..].to_string();
            tx.send((request, body)).expect("send captured request");

            let response_body = r#"{"choices":[{"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"search","arguments":"{\"q\":1}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":9,"completion_tokens":4}}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            socket.write_all(response.as_bytes()).await.unwrap();
        });

        let provider = OpenCodeGoProvider::new(loopback_config(addr), 5).unwrap();
        let prior_call = ToolCall {
            id: "call_0".to_string(),
            name: "search".to_string(),
            arguments: serde_json::json!({"q": 0}),
            reasoning: None,
            signature: None,
            arguments_parse_error: None,
        };
        let mut metadata = HashMap::new();
        metadata.insert("session_id".to_string(), "thread-a".to_string());
        let request = ToolCompletionRequest::new(
            vec![
                crate::provider::ChatMessage::user_with_parts(
                    "find it",
                    vec![ContentPart::ImageUrl {
                        image_url: ImageUrl {
                            url: "https://example.com/pic.png".to_string(),
                            detail: None,
                        },
                    }],
                ),
                crate::provider::ChatMessage::assistant_with_tool_calls(None, vec![prior_call])
                    .with_reasoning(Some("prior thought".to_string())),
                crate::provider::ChatMessage::tool_result("call_0", "search", "done"),
            ],
            vec![ToolDefinition {
                name: "search".to_string(),
                description: "Search".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": { "q": { "type": "integer" } }
                }),
            }],
        )
        .with_max_tokens(32)
        .with_temperature(0.2)
        .with_stop_sequences(vec!["END".to_string()])
        .with_tool_choice("required");
        let request = ToolCompletionRequest {
            response_format: Some(CompletionResponseFormat::JsonObject),
            metadata,
            ..request
        };

        let response = provider.complete_with_tools(request).await.unwrap();
        assert_eq!(response.finish_reason, FinishReason::ToolUse);
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].name, "search");
        assert_eq!(
            response.tool_calls[0].arguments,
            serde_json::json!({"q": 1})
        );

        let (raw_request, body) = rx.await.expect("captured request");
        let expected_session = session_header_value("thread-a");
        assert!(
            raw_request.contains(&format!("x-opencode-session: {expected_session}"))
                || raw_request.contains(&format!("X-Opencode-Session: {expected_session}"))
        );

        let payload: serde_json::Value = serde_json::from_str(&body).unwrap();
        let serialized = serde_json::to_string(&payload).unwrap();
        assert!(serialized.contains("\"tool_calls\""));
        assert!(serialized.contains("\"tool_call_id\":\"call_0\""));
        assert!(serialized.contains("\"reasoning_content\":\"prior thought\""));
        assert!(serialized.contains("\"image_url\""));
        assert!(serialized.contains("\"max_tokens\":32"));
        assert!(serialized.contains("\"temperature\":0.2"));
        assert!(serialized.contains("\"stop\":[\"END\"]"));
        assert!(serialized.contains("\"response_format\":{\"type\":\"json_object\"}"));
        assert!(serialized.contains("\"tool_choice\":\"required\""));
    }
}
