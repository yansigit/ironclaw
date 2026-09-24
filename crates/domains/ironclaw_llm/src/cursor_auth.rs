//! Cursor PKCE login codec: URL construction and poll/refresh body parsing.

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use rand::RngExt as _;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use uuid::Uuid;

pub const CURSOR_REFRESH_URL: &str = "https://api2.cursor.sh/auth/exchange_user_api_key";

const TOKEN_REFRESH_BUFFER_MS: i64 = 5 * 60 * 1000;
const FALLBACK_TTL_MS: i64 = 60 * 60 * 1000;

pub struct CursorAuthParams {
    pub verifier: String,
    pub challenge: String,
    pub uuid: String,
    pub login_url: String,
}

impl std::fmt::Debug for CursorAuthParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CursorAuthParams")
            .field("verifier", &"[REDACTED]")
            .field("challenge", &self.challenge)
            .field("uuid", &self.uuid)
            .field("login_url", &self.login_url)
            .finish()
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct CursorTokens {
    pub access_token: String,
    pub refresh_token: String,
    pub expires_at_ms: i64,
}

impl std::fmt::Debug for CursorTokens {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CursorTokens")
            .field("access_token", &"[REDACTED]")
            .field("refresh_token", &"[REDACTED]")
            .field("expires_at_ms", &self.expires_at_ms)
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PollParse {
    Pending,
    Ready(CursorTokens),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PollError {
    Terminal { status: u16 },
    Malformed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RefreshError {
    Retryable { status: u16 },
    Rejected { status: u16 },
}

#[derive(Debug, Deserialize)]
struct TokenResponse {
    #[serde(rename = "accessToken")]
    access_token: Option<String>,
    #[serde(rename = "refreshToken")]
    refresh_token: Option<String>,
}

pub fn generate_cursor_auth_params() -> CursorAuthParams {
    let mut rng = rand::rng();
    let verifier_bytes: [u8; 32] = rng.random();
    let verifier = URL_SAFE_NO_PAD.encode(verifier_bytes);
    let challenge = sha256_base64url(&verifier);
    let uuid = Uuid::new_v4().to_string();
    let login_url = cursor_login_url(&challenge, &uuid);
    CursorAuthParams {
        verifier,
        challenge,
        uuid,
        login_url,
    }
}

pub fn cursor_login_url(challenge: &str, uuid: &str) -> String {
    format!(
        "https://cursor.com/loginDeepControl?challenge={challenge}&uuid={uuid}&mode=login&redirectTarget=cli"
    )
}

pub fn cursor_poll_url(uuid: &str, verifier: &str) -> String {
    format!("https://api2.cursor.sh/auth/poll?uuid={uuid}&verifier={verifier}")
}

pub fn credentials_from_cursor_tokens(access_token: &str, refresh_token: &str) -> CursorTokens {
    let expires_at_ms = jwt_exp_ms(access_token)
        .map(|exp_ms| exp_ms - TOKEN_REFRESH_BUFFER_MS)
        .unwrap_or_else(fallback_expires_ms);
    CursorTokens {
        access_token: access_token.to_string(),
        refresh_token: refresh_token.to_string(),
        expires_at_ms,
    }
}

pub fn parse_poll_body(status: u16, body: &str) -> Result<PollParse, PollError> {
    if status == 404 {
        return Ok(PollParse::Pending);
    }
    if matches!(status, 400 | 401 | 403 | 410) {
        return Err(PollError::Terminal { status });
    }
    if status == 200 {
        let parsed: TokenResponse = serde_json::from_str(body).map_err(|_| PollError::Malformed)?;
        let access = parsed.access_token.as_deref();
        let refresh = parsed.refresh_token.as_deref();
        match (access, refresh) {
            (Some(access_token), Some(refresh_token)) => Ok(PollParse::Ready(
                credentials_from_cursor_tokens(access_token, refresh_token),
            )),
            _ => Err(PollError::Malformed),
        }
    } else {
        Err(PollError::Malformed)
    }
}

pub fn parse_refresh_body(
    status: u16,
    body: &str,
    previous_refresh: &str,
) -> Result<CursorTokens, RefreshError> {
    if matches!(status, 429 | 500 | 502 | 503 | 504) {
        return Err(RefreshError::Retryable { status });
    }
    if status != 200 {
        return Err(RefreshError::Rejected { status });
    }
    let parsed: TokenResponse = serde_json::from_str(body).map_err(|_| {
        RefreshError::Rejected {
            status: status,
        }
    })?;
    let access_token = parsed
        .access_token
        .filter(|t| !t.is_empty())
        .ok_or(RefreshError::Rejected { status })?;
    let refresh_token = parsed
        .refresh_token
        .filter(|t| !t.is_empty())
        .unwrap_or_else(|| previous_refresh.to_string());
    Ok(credentials_from_cursor_tokens(&access_token, &refresh_token))
}

fn sha256_base64url(verifier: &str) -> String {
    URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()))
}

fn jwt_exp_ms(access_token: &str) -> Option<i64> {
    let parts: Vec<&str> = access_token.split('.').collect();
    if parts.len() < 2 {
        return None;
    }
    let payload_b64 = parts[1];
    let decoded = URL_SAFE_NO_PAD.decode(payload_b64).or_else(|_| {
        let padded = pad_base64url(payload_b64);
        URL_SAFE_NO_PAD.decode(padded)
    });
    let decoded = decoded.ok()?;
    let payload: serde_json::Value = serde_json::from_slice(&decoded).ok()?;
    payload
        .get("exp")
        .and_then(|v| v.as_i64())
        .map(|exp_secs| exp_secs * 1000)
}

fn pad_base64url(input: &str) -> String {
    let rem = input.len() % 4;
    if rem == 0 {
        return input.to_string();
    }
    let pad = 4 - rem;
    format!("{}{}", input, "=".repeat(pad))
}

fn fallback_expires_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64 + FALLBACK_TTL_MS)
        .unwrap_or(FALLBACK_TTL_MS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine;

    fn sha256_base64url(verifier: &str) -> String {
        URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()))
    }

    #[test]
    fn login_url_carries_challenge_not_verifier() {
        let params = generate_cursor_auth_params();
        assert!(params.login_url.starts_with("https://cursor.com/loginDeepControl?"));
        assert!(params.login_url.contains(&format!("challenge={}", params.challenge)));
        assert!(params.login_url.contains(&format!("uuid={}", params.uuid)));
        assert!(params.login_url.contains("mode=login"));
        assert!(params.login_url.contains("redirectTarget=cli"));
        assert!(!params.login_url.contains(&params.verifier));
        assert_eq!(params.challenge, sha256_base64url(&params.verifier));
    }

    #[test]
    fn poll_404_is_pending_and_200_returns_tokens() {
        let pending = parse_poll_body(404, "").expect("404");
        assert!(matches!(pending, PollParse::Pending));
        let ready = parse_poll_body(
            200,
            r#"{"accessToken":"a.e30.x","refreshToken":"r"}"#,
        )
        .expect("200");
        match ready {
            PollParse::Ready(tokens) => {
                assert_eq!(tokens.access_token, "a.e30.x");
                assert_eq!(tokens.refresh_token, "r");
            }
            PollParse::Pending => panic!("expected tokens"),
        }
    }

    #[test]
    fn poll_403_is_terminal() {
        let err = parse_poll_body(403, "").expect_err("terminal");
        assert!(matches!(err, PollError::Terminal { status: 403 }));
    }

    #[test]
    fn refresh_keeps_previous_when_server_omits_refresh() {
        let tokens = parse_refresh_body(200, r#"{"accessToken":"next"}"#, "old-refresh")
            .expect("ok");
        assert_eq!(tokens.access_token, "next");
        assert_eq!(tokens.refresh_token, "old-refresh");
    }

    #[test]
    fn jwt_exp_is_skewed_five_minutes() {
        let payload =
            base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(r#"{"exp":2000000000}"#);
        let access = format!("h.{payload}.s");
        let tokens = credentials_from_cursor_tokens(&access, "r");
        assert_eq!(tokens.expires_at_ms, 2000000000 * 1000 - 5 * 60 * 1000);
    }
}
