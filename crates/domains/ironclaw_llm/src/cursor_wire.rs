//! Cursor AgentService connect/protobuf wire codec.

use std::collections::HashMap;

use sha2::{Digest, Sha256};

pub const LOCAL_TOOL_REJECTION: &str = "IronClaw does not execute Cursor local tools";

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum WireError {
    #[error("incomplete frame")]
    Incomplete,
    #[error("invalid wire data")]
    Invalid,
}

pub fn encode_connect_frame(payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 + 4 + payload.len());
    out.push(0x00);
    let len = payload.len() as u32;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(payload);
    out
}

pub fn decode_connect_frames(input: &[u8]) -> Result<Vec<Vec<u8>>, WireError> {
    let mut frames = Vec::new();
    let mut offset = 0;
    while offset < input.len() {
        if offset + 5 > input.len() {
            return Err(WireError::Incomplete);
        }
        let flag = input[offset];
        offset += 1;
        let len = u32::from_be_bytes([
            input[offset],
            input[offset + 1],
            input[offset + 2],
            input[offset + 3],
        ]) as usize;
        offset += 4;
        if offset + len > input.len() {
            return Err(WireError::Incomplete);
        }
        let payload = &input[offset..offset + len];
        offset += len;
        let decoded = match flag {
            0x00 => payload.to_vec(),
            0x01 => {
                use std::io::Read;
                let mut decoder = flate2::read::GzDecoder::new(payload);
                let mut decompressed = Vec::new();
                decoder
                    .read_to_end(&mut decompressed)
                    .map_err(|_| WireError::Invalid)?;
                decompressed
            }
            _ => payload.to_vec(),
        };
        frames.push(decoded);
    }
    Ok(frames)
}

pub fn normalize_cursor_model_id(model_id: &str) -> String {
    let trimmed = model_id.trim();
    let lower = trimmed.to_ascii_lowercase();
    match lower.as_str() {
        "" | "composer-2-5" | "composer-2.5-sdk" | "composer-latest" => {
            "composer-2.5".to_string()
        }
        "composer-2-5-fast" | "composer-2.5-sdk-fast" | "composer-latest-fast" => {
            "composer-2.5-fast".to_string()
        }
        _ => trimmed.to_string(),
    }
}

pub fn resolve_requested_model(model_id: &str) -> (String, Vec<(String, String)>) {
    let trimmed = model_id.trim();
    if trimmed.eq_ignore_ascii_case("auto") {
        return ("default".to_string(), Vec::new());
    }
    if trimmed.eq_ignore_ascii_case("composer-2.5-fast") {
        return (
            "composer-2.5".to_string(),
            vec![("fast".to_string(), "true".to_string())],
        );
    }
    (normalize_cursor_model_id(trimmed), Vec::new())
}

pub struct AgentRunInput {
    pub model_id: String,
    pub user_text: String,
    pub conversation_id: String,
    pub message_id: String,
    pub system_prompt: Option<String>,
}

pub struct EncodedRun {
    pub frame: Vec<u8>,
    pub blobs: HashMap<String, Vec<u8>>,
}

pub fn encode_agent_run(input: &AgentRunInput) -> EncodedRun {
    let (resolved_model, params) = resolve_requested_model(&input.model_id);
    let mut blobs = HashMap::new();

    let conversation_state = if let Some(prompt) = &input.system_prompt {
        let json = format!(r#"{{"role":"system","content":{}}}"#, json_string_escape(prompt));
        let digest = Sha256::digest(json.as_bytes());
        let hex_key = hex::encode(digest);
        blobs.insert(hex_key, json.into_bytes());
        let state_inner = encode_bytes_field(1, &digest);
        encode_message_field(1, &state_inner)
    } else {
        Vec::new()
    };

    let user_message = concat_fields(&[
        encode_string_field(1, &input.user_text),
        encode_string_field(2, &input.message_id),
        encode_message_field(3, &[]),
        encode_varint_field(4, 1),
    ]);
    let user_message_action = encode_message_field(1, &user_message);
    let action = encode_message_field(1, &user_message_action);

    let model_details = concat_fields(&[
        encode_string_field(1, &resolved_model),
        encode_string_field(3, &resolved_model),
        encode_string_field(4, &resolved_model),
    ]);

    let mcp_tools = encode_message_field(4, &encode_message_field(1, &[]));

    let mut requested_model_msg = encode_string_field(1, &resolved_model);
    for (id, value) in &params {
        let param = concat_fields(&[
            encode_string_field(1, id),
            encode_string_field(2, value),
        ]);
        requested_model_msg.extend(encode_message_field(3, &param));
    }

    let run_request = concat_fields(&[
        conversation_state,
        encode_message_field(2, &action),
        encode_message_field(3, &model_details),
        mcp_tools,
        encode_string_field(5, &input.conversation_id),
        encode_message_field(9, &requested_model_msg),
        encode_varint_field(12, 0),
        encode_string_field(16, &input.message_id),
    ]);

    let client_msg = encode_message_field(1, &run_request);
    EncodedRun {
        frame: encode_connect_frame(&client_msg),
        blobs,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServerEvent {
    Text { text: String },
    Thinking { text: String },
    TurnEnded,
    Heartbeat,
    Exec(ExecEvent),
    Kv(KvEvent),
    Ignore,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecEvent {
    RequestContext {
        exec_msg_id: u64,
        exec_id: String,
    },
    Shell {
        exec_msg_id: u64,
        exec_id: String,
        command: String,
        working_dir: String,
    },
    Other {
        exec_msg_id: u64,
        exec_id: String,
        variant_field: u32,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvEvent {
    GetBlob {
        kv_id: u64,
        blob_id: Vec<u8>,
        request_metadata: Option<Vec<u8>>,
    },
    SetBlob {
        kv_id: u64,
        request_metadata: Option<Vec<u8>>,
    },
}

pub fn decode_server_payload(payload: &[u8]) -> Vec<ServerEvent> {
    let mut events = Vec::new();
    for (field, wire, value) in parse_fields(payload) {
        match field {
            1 if wire == 2 => {
                if let Some(ev) = decode_interaction_update(value) {
                    events.push(ev);
                }
            }
            2 if wire == 2 => {
                if let Some(ev) = decode_exec_server_message(value) {
                    events.push(ServerEvent::Exec(ev));
                }
            }
            4 if wire == 2 => {
                if let Some(ev) = decode_kv_server_message(value) {
                    events.push(ServerEvent::Kv(ev));
                }
            }
            _ => {}
        }
    }
    events
}

fn decode_interaction_update(msg: &[u8]) -> Option<ServerEvent> {
    for (field, wire, value) in parse_fields(msg) {
        match field {
            1 if wire == 2 => {
                let text = protobuf_string_field(value, 1)?;
                return Some(ServerEvent::Text { text });
            }
            4 if wire == 2 => {
                let text = protobuf_string_field(value, 1)?;
                return Some(ServerEvent::Thinking { text });
            }
            13 if wire == 0 => return Some(ServerEvent::Heartbeat),
            14 if wire == 0 => return Some(ServerEvent::TurnEnded),
            _ => {}
        }
    }
    Some(ServerEvent::Ignore)
}

fn decode_exec_server_message(msg: &[u8]) -> Option<ExecEvent> {
    let mut exec_msg_id = 0u64;
    let mut exec_id = String::new();
    let mut shell_args: Option<&[u8]> = None;
    let mut request_context: Option<&[u8]> = None;
    let mut other_variant: Option<u32> = None;

    for (field, wire, value) in parse_fields(msg) {
        match field {
            1 if wire == 0 => exec_msg_id = decode_varint(value).unwrap_or(0),
            15 if wire == 2 => exec_id = String::from_utf8_lossy(value).into_owned(),
            2 if wire == 2 => shell_args = Some(value),
            10 if wire == 2 => request_context = Some(value),
            f if wire == 2 && f != 2 && f != 10 && f != 15 => other_variant = Some(f),
            _ => {}
        }
    }

    if let Some(args) = shell_args {
        let command = protobuf_string_field(args, 1).unwrap_or_default();
        let working_dir = protobuf_string_field(args, 2).unwrap_or_default();
        return Some(ExecEvent::Shell {
            exec_msg_id,
            exec_id,
            command,
            working_dir,
        });
    }
    if request_context.is_some() {
        return Some(ExecEvent::RequestContext {
            exec_msg_id,
            exec_id,
        });
    }
    if let Some(vf) = other_variant {
        return Some(ExecEvent::Other {
            exec_msg_id,
            exec_id,
            variant_field: vf,
        });
    }
    None
}

fn decode_kv_server_message(msg: &[u8]) -> Option<KvEvent> {
    let mut kv_id = 0u64;
    let mut request_metadata: Option<Vec<u8>> = None;
    let mut get_blob: Option<&[u8]> = None;
    let mut set_blob = false;

    for (field, wire, value) in parse_fields(msg) {
        match field {
            1 if wire == 0 => kv_id = decode_varint(value).unwrap_or(0),
            2 if wire == 2 => get_blob = Some(value),
            3 if wire == 2 => set_blob = true,
            4 if wire == 2 => request_metadata = Some(value.to_vec()),
            _ => {}
        }
    }

    if let Some(args) = get_blob {
        let blob_id = protobuf_bytes_field(args, 1).unwrap_or_default();
        return Some(KvEvent::GetBlob {
            kv_id,
            blob_id,
            request_metadata,
        });
    }
    if set_blob {
        return Some(KvEvent::SetBlob {
            kv_id,
            request_metadata,
        });
    }
    None
}

pub fn encode_request_context_response(exec_msg_id: u64, exec_id: &str) -> Vec<u8> {
    let success = encode_message_field(1, &[]);
    let request_context_result = concat_fields(&[
        encode_message_field(1, &success),
    ]);
    let exec_client = concat_fields(&[
        encode_varint_field(1, exec_msg_id),
        encode_string_field(15, exec_id),
        encode_message_field(10, &request_context_result),
    ]);
    let client_msg = encode_message_field(2, &exec_client);
    encode_connect_frame(&client_msg)
}

pub fn encode_shell_rejected(
    exec_msg_id: u64,
    exec_id: &str,
    command: &str,
    working_dir: &str,
) -> Vec<u8> {
    let rejected = concat_fields(&[
        encode_string_field(1, command),
        encode_string_field(2, working_dir),
        encode_string_field(3, LOCAL_TOOL_REJECTION),
    ]);
    let shell_result = encode_message_field(2, &encode_message_field(2, &rejected));
    let exec_client = concat_fields(&[
        encode_varint_field(1, exec_msg_id),
        encode_string_field(15, exec_id),
        shell_result,
    ]);
    let client_msg = encode_message_field(2, &exec_client);
    encode_connect_frame(&client_msg)
}

pub fn encode_exec_rejected(exec_msg_id: u64, exec_id: &str, variant_field: u32) -> Vec<u8> {
    let inner = encode_string_field(1, LOCAL_TOOL_REJECTION);
    let variant_msg = encode_message_field(2, &inner);
    let exec_client = concat_fields(&[
        encode_varint_field(1, exec_msg_id),
        encode_string_field(15, exec_id),
        encode_message_field(variant_field, &variant_msg),
    ]);
    let client_msg = encode_message_field(2, &exec_client);
    encode_connect_frame(&client_msg)
}

pub fn encode_kv_get_result(
    kv_id: u64,
    blob_data: &[u8],
    request_metadata: Option<&[u8]>,
) -> Vec<u8> {
    let get_blob_result = encode_bytes_field(1, blob_data);
    let mut kv_client = concat_fields(&[
        encode_varint_field(1, kv_id),
        encode_message_field(2, &get_blob_result),
    ]);
    if let Some(meta) = request_metadata {
        kv_client.extend(encode_bytes_field(4, meta));
    }
    let client_msg = encode_message_field(3, &kv_client);
    encode_connect_frame(&client_msg)
}

pub fn encode_kv_set_result(kv_id: u64, request_metadata: Option<&[u8]>) -> Vec<u8> {
    let set_blob_result = Vec::new();
    let mut kv_client = concat_fields(&[
        encode_varint_field(1, kv_id),
        encode_message_field(3, &set_blob_result),
    ]);
    if let Some(meta) = request_metadata {
        kv_client.extend(encode_bytes_field(4, meta));
    }
    let client_msg = encode_message_field(3, &kv_client);
    encode_connect_frame(&client_msg)
}

pub fn visible_composer_text(thinking: &str) -> String {
    const MARKER: &str = "</think>";
    if let Some(pos) = thinking.rfind(MARKER) {
        let mut visible = thinking[pos + MARKER.len()..].trim().to_string();
        for prefix in ["<｜final｜>", "<|final|>"] {
            if visible.starts_with(prefix) {
                visible = visible[prefix.len()..].trim_start().to_string();
            }
        }
        for suffix in ["<｜/final｜>", "<|/final|>"] {
            if visible.ends_with(suffix) {
                let new_len = visible.len() - suffix.len();
                visible = visible[..new_len].trim_end().to_string();
            }
        }
        visible
    } else {
        String::new()
    }
}

pub(crate) fn encode_shell_server_frame(
    exec_msg_id: u64,
    exec_id: &str,
    command: &str,
    working_dir: &str,
) -> Vec<u8> {
    let shell_args = concat_fields(&[
        encode_string_field(1, command),
        encode_string_field(2, working_dir),
    ]);
    let exec_server = concat_fields(&[
        encode_varint_field(1, exec_msg_id),
        encode_message_field(2, &shell_args),
        encode_string_field(15, exec_id),
    ]);
    let server_msg = encode_message_field(2, &exec_server);
    encode_connect_frame(&server_msg)
}

// --- protobuf helpers ---

fn concat_fields(parts: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    for p in parts {
        out.extend_from_slice(p);
    }
    out
}

fn encode_key(field_number: u32, wire_type: u32) -> Vec<u8> {
    encode_varint(((field_number as u64) << 3) | wire_type as u64)
}

fn encode_varint(mut value: u64) -> Vec<u8> {
    let mut out = Vec::new();
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
    out
}

fn decode_varint(buf: &[u8]) -> Option<u64> {
    let mut result = 0u64;
    let mut shift = 0;
    for &byte in buf {
        result |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 {
            return Some(result);
        }
        shift += 7;
        if shift >= 64 {
            return None;
        }
    }
    None
}

fn encode_varint_field(field_number: u32, value: u64) -> Vec<u8> {
    let mut out = encode_key(field_number, 0);
    out.extend(encode_varint(value));
    out
}

fn encode_string_field(field_number: u32, s: &str) -> Vec<u8> {
    encode_bytes_field(field_number, s.as_bytes())
}

fn encode_bytes_field(field_number: u32, data: &[u8]) -> Vec<u8> {
    let mut out = encode_key(field_number, 2);
    out.extend(encode_varint(data.len() as u64));
    out.extend_from_slice(data);
    out
}

fn encode_message_field(field_number: u32, msg: &[u8]) -> Vec<u8> {
    encode_bytes_field(field_number, msg)
}

fn parse_fields(payload: &[u8]) -> Vec<(u32, u32, &[u8])> {
    let mut fields = Vec::new();
    let mut offset = 0;
    while offset < payload.len() {
        let start = offset;
        let key_bytes = read_varint_slice(payload, &mut offset);
        if key_bytes.is_none() {
            break;
        }
        let key = decode_varint(key_bytes.unwrap()).unwrap_or(0);
        let field_number = (key >> 3) as u32;
        let wire_type = (key & 7) as u32;
        let value = match wire_type {
            0 => {
                let v = read_varint_slice(payload, &mut offset).unwrap_or(&[]);
                v
            }
            2 => {
                let len = decode_varint(read_varint_slice(payload, &mut offset).unwrap_or(&[]))
                    .unwrap_or(0) as usize;
                let vstart = offset;
                if offset + len > payload.len() {
                    break;
                }
                offset += len;
                &payload[vstart..offset]
            }
            1 => {
                if offset + 8 > payload.len() {
                    break;
                }
                offset += 8;
                &payload[start..offset]
            }
            5 => {
                if offset + 4 > payload.len() {
                    break;
                }
                offset += 4;
                &payload[start..offset]
            }
            _ => break,
        };
        if wire_type == 0 {
            fields.push((field_number, wire_type, value));
        } else if wire_type == 2 {
            fields.push((field_number, wire_type, value));
        } else {
            fields.push((field_number, wire_type, value));
        }
    }
    fields
}

fn read_varint_slice<'a>(payload: &'a [u8], offset: &mut usize) -> Option<&'a [u8]> {
    let start = *offset;
    while *offset < payload.len() {
        if payload[*offset] & 0x80 == 0 {
            *offset += 1;
            return Some(&payload[start..*offset]);
        }
        *offset += 1;
    }
    if start < payload.len() {
        Some(&payload[start..])
    } else {
        None
    }
}

fn protobuf_string_field(msg: &[u8], field_number: u32) -> Option<String> {
    for (f, wire, value) in parse_fields(msg) {
        if f == field_number && wire == 2 {
            return Some(String::from_utf8_lossy(value).into_owned());
        }
    }
    None
}

fn protobuf_bytes_field(msg: &[u8], field_number: u32) -> Option<Vec<u8>> {
    for (f, wire, value) in parse_fields(msg) {
        if f == field_number && wire == 2 {
            return Some(value.to_vec());
        }
    }
    None
}

fn json_string_escape(s: &str) -> String {
    serde_json::Value::String(s.to_string()).to_string()
}

pub(crate) fn protobuf_string_path(payload: &[u8], path: &[u32]) -> Option<String> {
    let mut cur = payload;
    for (i, &field_num) in path.iter().enumerate() {
        let is_last = i == path.len() - 1;
        let mut found = None;
        for (f, wire, value) in parse_fields(cur) {
            if f == field_num && wire == 2 {
                found = Some(value);
                break;
            }
        }
        let value = found?;
        if is_last {
            return Some(String::from_utf8_lossy(value).into_owned());
        }
        cur = value;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn protobuf_string_path(payload: &[u8], path: &[u32]) -> Option<String> {
        super::protobuf_string_path(payload, path)
    }

    fn protobuf_message_path<'a>(payload: &'a [u8], path: &[u32]) -> Option<&'a [u8]> {
        let mut cur = payload;
        for (i, &field_num) in path.iter().enumerate() {
            let is_last = i == path.len() - 1;
            let mut found = None;
            for (f, wire, value) in parse_fields(cur) {
                if f == field_num && wire == 2 {
                    found = Some(value);
                    break;
                }
            }
            let value = found?;
            if is_last {
                return Some(value);
            }
            cur = value;
        }
        None
    }

    #[test]
    fn composer_aliases_normalize() {
        assert_eq!(normalize_cursor_model_id("composer-2-5"), "composer-2.5");
        assert_eq!(normalize_cursor_model_id(""), "composer-2.5");
        assert_eq!(
            normalize_cursor_model_id("composer-latest-fast"),
            "composer-2.5-fast"
        );
        assert_eq!(normalize_cursor_model_id("gpt-5.5"), "gpt-5.5");
    }

    #[test]
    fn fast_composer_splits_parameter() {
        let (id, params) = resolve_requested_model("composer-2.5-fast");
        assert_eq!(id, "composer-2.5");
        assert_eq!(params, vec![("fast".to_string(), "true".to_string())]);
    }

    #[test]
    fn run_frame_round_trips_user_text_and_model() {
        let encoded = encode_agent_run(&AgentRunInput {
            model_id: "composer-2.5".into(),
            user_text: "ping".into(),
            conversation_id: "conv-1".into(),
            message_id: "msg-1".into(),
            system_prompt: None,
        });
        let frames = decode_connect_frames(&encoded.frame).expect("frame");
        assert_eq!(frames.len(), 1);
        let text = protobuf_string_path(&frames[0], &[1, 2, 1, 1, 1]);
        assert_eq!(text.as_deref(), Some("ping"));
        let model = protobuf_string_path(&frames[0], &[1, 9, 1]);
        assert_eq!(model.as_deref(), Some("composer-2.5"));
        let mcp_tools = protobuf_message_path(&frames[0], &[1, 4]).expect("mcp_tools field 4");
        assert!(
            parse_fields(mcp_tools)
                .iter()
                .any(|(f, wire, _)| *f == 1 && *wire == 2),
            "McpTools wrapper must contain inner field 1"
        );
    }

    #[test]
    fn shell_rejection_does_not_echo_a_successful_result() {
        let frame = encode_shell_rejected(7, "exec-9", "rm -rf /", "/tmp");
        let frames = decode_connect_frames(&frame).expect("frame");
        let reason = protobuf_string_path(&frames[0], &[2, 2, 2, 3]);
        assert_eq!(reason.as_deref(), Some(LOCAL_TOOL_REJECTION));
        let blob = String::from_utf8_lossy(&frames[0]);
        assert!(!blob.contains("rm -rf / exited"));
    }

    #[test]
    fn composer_visible_text_drops_thinking() {
        let visible = visible_composer_text("secret chain</think>\nhello");
        assert_eq!(visible, "hello");
        assert!(visible_composer_text("still thinking").is_empty());
    }

    #[test]
    fn shell_server_frame_decodes_as_shell_exec() {
        let frame = encode_shell_server_frame(4, "e1", "uname", "/tmp");
        let payloads = decode_connect_frames(&frame).expect("frame");
        let events = decode_server_payload(&payloads[0]);
        assert!(matches!(
            events.first(),
            Some(ServerEvent::Exec(ExecEvent::Shell {
                exec_msg_id: 4,
                exec_id,
                command,
                working_dir,
            })) if exec_id == "e1" && command == "uname" && working_dir == "/tmp"
        ));
    }

    #[test]
    fn system_prompt_is_addressed_by_sha256() {
        let encoded = encode_agent_run(&AgentRunInput {
            model_id: "composer-2.5".into(),
            user_text: "ping".into(),
            conversation_id: "conv-1".into(),
            message_id: "msg-1".into(),
            system_prompt: Some("be brief".into()),
        });
        assert_eq!(encoded.blobs.len(), 1);
        let (_hex, bytes) = encoded.blobs.iter().next().unwrap();
        assert_eq!(bytes, br#"{"role":"system","content":"be brief"}"#);
    }
}
