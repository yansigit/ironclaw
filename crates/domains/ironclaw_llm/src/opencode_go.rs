use std::time::Duration;

use async_trait::async_trait;
use rust_decimal::Decimal;
use sha2::{Digest, Sha256};

use crate::config::OpenCodeGoConfig;
use crate::error::LlmError;
use crate::provider::{
    map_provider_finish_token, ChatMessage, CompletionRequest, CompletionResponse, FinishReason,
    LlmProvider, Role, ToolCall, ToolCompletionRequest, ToolCompletionResponse, ToolDefinition,
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
            "content": message.content,
        });
        if let Some(tool_call_id) = &message.tool_call_id {
            value["tool_call_id"] = serde_json::Value::String(tool_call_id.clone());
        }
        value
    }

    fn body(&self, model: &str, messages: &[ChatMessage], tools: &[ToolDefinition]) -> serde_json::Value {
        let mut body = serde_json::json!({
            "model": model,
            "messages": messages.iter().map(Self::message_json).collect::<Vec<_>>(),
        });
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
        body
    }

    async fn post_chat(
        &self,
        model: &str,
        messages: &[ChatMessage],
        tools: &[ToolDefinition],
        lane: &str,
    ) -> Result<serde_json::Value, LlmError> {
        if self.config.api_key.trim().is_empty() {
            return Err(LlmError::AuthFailed {
                provider: "opencode_go".to_string(),
            });
        }
        if matches!(wire_for_model(model), OpenCodeGoWire::Responses) {
            return Err(LlmError::RequestFailed {
                provider: "opencode_go".to_string(),
                reason: format!(
                    "model {model} uses the OpenCode Go responses wire, which this provider does not send"
                ),
            });
        }
        let url = format!("{}/chat/completions", self.config.base_url.trim_end_matches('/'));
        let response = self
            .http
            .post(url)
            .bearer_auth(&self.config.api_key)
            .header(SESSION_HEADER, session_header_value(lane))
            .json(&self.body(model, messages, tools))
            .send()
            .await
            .map_err(|error| LlmError::RequestFailed {
                provider: "opencode_go".to_string(),
                reason: error.to_string(),
            })?;
        let status = response.status();
        let text = response.text().await.map_err(|error| LlmError::RequestFailed {
            provider: "opencode_go".to_string(),
            reason: error.to_string(),
        })?;
        if !status.is_success() {
            let mut reason = text;
            reason.truncate(512);
            return Err(LlmError::RequestFailed {
                provider: "opencode_go".to_string(),
                reason: format!("HTTP {status}: {reason}"),
            });
        }
        serde_json::from_str(&text).map_err(|error| LlmError::RequestFailed {
            provider: "opencode_go".to_string(),
            reason: error.to_string(),
        })
    }

    async fn fetch_model_ids(&self) -> Result<Vec<String>, LlmError> {
        if self.config.api_key.trim().is_empty() {
            return Err(LlmError::AuthFailed {
                provider: "opencode_go".to_string(),
            });
        }
        let url = format!("{}/models", self.config.base_url.trim_end_matches('/'));
        let response = self
            .http
            .get(url)
            .bearer_auth(&self.config.api_key)
            .header(SESSION_HEADER, session_header_value("ironclaw"))
            .send()
            .await
            .map_err(|error| LlmError::RequestFailed {
                provider: "opencode_go".to_string(),
                reason: error.to_string(),
            })?;
        let status = response.status();
        let text = response.text().await.map_err(|error| LlmError::RequestFailed {
            provider: "opencode_go".to_string(),
            reason: error.to_string(),
        })?;
        if !status.is_success() {
            let mut reason = text;
            reason.truncate(512);
            return Err(LlmError::RequestFailed {
                provider: "opencode_go".to_string(),
                reason: format!("HTTP {status}: {reason}"),
            });
        }
        let payload: serde_json::Value = serde_json::from_str(&text).map_err(|error| {
            LlmError::RequestFailed {
                provider: "opencode_go".to_string(),
                reason: error.to_string(),
            }
        })?;
        let Some(data) = payload["data"].as_array() else {
            return Err(LlmError::RequestFailed {
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
        let payload = self
            .post_chat(&model, &request.messages, &[], &self.lane_from_metadata(&request.metadata))
            .await?;
        let choice = &payload["choices"][0];
        let content = choice["message"]["content"].as_str().unwrap_or("").to_string();
        let finish_reason = choice["finish_reason"]
            .as_str()
            .and_then(map_provider_finish_token)
            .unwrap_or(FinishReason::Unknown);
        Ok(CompletionResponse {
            content,
            input_tokens: payload["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32,
            output_tokens: payload["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32,
            finish_reason,
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
            .clone()
            .unwrap_or_else(|| self.config.model.clone());
        let payload = self
            .post_chat(
                &model,
                &request.messages,
                &request.tools,
                &self.lane_from_metadata(&request.metadata),
            )
            .await?;
        let choice = &payload["choices"][0];
        let content = choice["message"]["content"].as_str().map(str::to_string);
        let mut tool_calls = Vec::new();
        if let Some(calls) = choice["message"]["tool_calls"].as_array() {
            for call in calls {
                let name = call["function"]["name"].as_str().unwrap_or("").to_string();
                let raw_args = call["function"]["arguments"].as_str().unwrap_or("{}");
                let (arguments, arguments_parse_error) = match serde_json::from_str::<serde_json::Value>(raw_args) {
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
        let finish_reason = choice["finish_reason"]
            .as_str()
            .and_then(map_provider_finish_token)
            .unwrap_or(FinishReason::Unknown);
        Ok(ToolCompletionResponse {
            content,
            tool_calls,
            input_tokens: payload["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32,
            output_tokens: payload["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32,
            finish_reason,
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 0,
            reasoning: None,
            reasoning_details: None,
        })
    }

    async fn list_models(&self) -> Result<Vec<String>, LlmError> {
        self.fetch_model_ids().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            assert!(request.contains("authorization: Bearer test-go-key") || request.contains("Authorization: Bearer test-go-key"));
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

        let provider = OpenCodeGoProvider::new(
            crate::config::OpenCodeGoConfig {
                model: "kimi-k2.7-code".to_string(),
                base_url: format!("http://{addr}"),
                api_key: "test-go-key".to_string(),
            },
            5,
        )
        .unwrap();
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

        let provider = OpenCodeGoProvider::new(
            crate::config::OpenCodeGoConfig {
                model: "kimi-k2.7-code".to_string(),
                base_url: format!("http://{addr}"),
                api_key: "test-go-key".to_string(),
            },
            5,
        )
        .unwrap();
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
            crate::config::OpenCodeGoConfig::build(None, None, Some(key)),
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
}
