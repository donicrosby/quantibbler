//! Chat-template application (qk-serve-bf16 BS-5, design S5).
//!
//! `/v1/chat/completions` carries an OpenAI `messages` array; the server must
//! render it into the single prompt string the model was trained on — the
//! comparison corpus for benchmarking is chat-format scenarios, so a drifted
//! prompt breaks prompt-token parity.
//!
//! The model's own Jinja2 `chat_template` (Qwen3) is the authority: it rides in
//! the artifact's `tokenizer.json` (`chat_template`) and is read at load —
//! the `tokenizers` crate does not expose arbitrary metadata fields.
//!
//! Rendering uses `minijinja` with a PINNED context (`add_generation_prompt =
//! true`, `enable_thinking = false`, design S5). The Qwen3 template calls
//! Python `str` methods (`content.startswith(...)`); minijinja ships no built-in
//! string methods, so [`python_str_method`] supplies exactly the ones the
//! template uses and refuses every other name — a template that needs an
//! unimplemented method fails loudly rather than rendering something subtly
//! different.

use minijinja::{Environment, Error as MjError, ErrorKind, State, Value, context};

use crate::api::types::ChatMessage;

/// The pinned render context (design S5): a generation prompt is always
/// appended and thinking is always off for this change.
const ADD_GENERATION_PROMPT: bool = true;
const ENABLE_THINKING: bool = false;

/// Render `messages` through `template` (BS-5). Any parse or render failure is
/// an `Err` — the caller turns it into a 400 (BS-5.2), never a silently
/// different prompt.
pub fn render(template: &str, messages: &[ChatMessage]) -> Result<String, String> {
    let mut env = Environment::new();
    env.set_unknown_method_callback(python_str_method);
    env.add_template("chat", template)
        .map_err(|e| format!("chat template is not renderable: {e}"))?;
    let tmpl = env
        .get_template("chat")
        .map_err(|e| format!("chat template lookup: {e}"))?;
    // The messages are handed over as plain maps (`role` / `content`), the two
    // fields the OpenAI chat shape and the template both name.
    let msgs: Value = messages
        .iter()
        .map(|m| {
            Value::from_pairs([
                ("role", Value::from(m.role.as_str())),
                ("content", Value::from(m.content.as_str())),
            ])
        })
        .collect();
    tmpl.render(context! {
        messages => msgs,
        add_generation_prompt => ADD_GENERATION_PROMPT,
        enable_thinking => ENABLE_THINKING,
    })
    .map_err(|e| format!("chat template render: {e}"))
}

/// The Python `str` methods the Qwen3 chat template calls. minijinja resolves
/// method calls on a string value through this hook; returning
/// `UnknownMethod` for every other name keeps the surface closed.
fn python_str_method(
    _state: &mut State,
    value: &Value,
    method: &str,
    args: &[Value],
) -> Result<Value, MjError> {
    let Some(s) = value.as_str() else {
        return Err(MjError::from(ErrorKind::UnknownMethod));
    };
    let arg =
        |i: usize| -> Option<String> { args.get(i).and_then(Value::as_str).map(str::to_string) };
    let seq = |parts: Vec<String>| -> Value { parts.into_iter().map(Value::from).collect() };
    match method {
        "startswith" => Ok(Value::from(s.starts_with(&arg(0).unwrap_or_default()))),
        "endswith" => Ok(Value::from(s.ends_with(&arg(0).unwrap_or_default()))),
        "split" => {
            let sep = arg(0).unwrap_or_default();
            if sep.is_empty() {
                return Err(MjError::from(ErrorKind::UnknownMethod));
            }
            Ok(seq(s.split(sep.as_str()).map(str::to_string).collect()))
        }
        // `strip` / `lstrip` / `rstrip`: with an argument the argument is a
        // CHARACTER SET (Python semantics), without one it is whitespace.
        "strip" | "lstrip" | "rstrip" => {
            let set = arg(0);
            let keep = |c: char| match &set {
                Some(chars) => !chars.contains(c),
                None => !c.is_whitespace(),
            };
            let out = match method {
                "strip" => s.trim_matches(keep),
                "lstrip" => s.trim_start_matches(keep),
                _ => s.trim_end_matches(keep),
            };
            Ok(Value::from(out))
        }
        _ => Err(MjError::from(ErrorKind::UnknownMethod)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.into(),
            content: content.into(),
        }
    }

    /// The pinned context renders a generation prompt and the thinking-off
    /// marker; the template's `str` method calls are served by the hook.
    #[test]
    fn renders_the_pinned_context() {
        let template = "{%- for m in messages %}{{ m.role }}:{{ m.content }};{% endfor %}\
                        {%- if add_generation_prompt %}|gen{% endif %}\
                        {%- if enable_thinking %}|think{% endif %}\
                        {{- '/' + messages[0].content.startswith('hel')|string }}";
        let got = render(template, &[msg("user", "hello")]).expect("render");
        assert_eq!(got, "user:hello;|gen/True");
    }

    /// An unimplemented method is a loud failure, not a silent empty render.
    #[test]
    fn unknown_string_method_is_an_error() {
        let err = render(
            "{{ messages[0].content.capitalize() }}",
            &[msg("user", "hi")],
        )
        .expect_err("an unimplemented method must not render");
        assert!(
            err.contains("unknown method") || err.contains("Unknown method"),
            "unexpected error: {err}"
        );
    }
}
