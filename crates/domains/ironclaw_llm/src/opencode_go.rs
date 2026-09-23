use sha2::{Digest, Sha256};

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
}
