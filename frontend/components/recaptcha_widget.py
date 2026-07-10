import streamlit as st

from auth.config import get_recaptcha_site_key
import streamlit.components.v2 as components

_RECAPTCHA_JS = """
export default function(component) {
    const { data, setStateValue } = component;
    const siteKey = data.site_key;
    const action = data.action || "submit";

    if (!siteKey || siteKey === "CHANGE_ME") {
        return;
    }

    const runCaptcha = () => {
        window.grecaptcha.execute(siteKey, { action })
            .then((token) => setStateValue("token", token))
            .catch(() => setStateValue("token", null));
    };

    const whenReady = () => {
        if (window.grecaptcha && window.grecaptcha.execute) {
            window.grecaptcha.ready(runCaptcha);
            return;
        }
        setTimeout(whenReady, 100);
    };

    if (!document.querySelector(`script[src*="recaptcha/api.js?render=${siteKey}"]`)) {
        const script = document.createElement("script");
        script.src = `https://www.google.com/recaptcha/api.js?render=${siteKey}`;
        script.async = true;
        document.head.appendChild(script);
    }

    whenReady();
}
"""

_recaptcha_component = components.component(
    "recaptcha_v3",
    css=".grecaptcha-badge { visibility: hidden; }",
    js=_RECAPTCHA_JS,
    isolate_styles=False,
)

_RECAPTCHA_DISCLAIMER = (
    "This site is protected by reCAPTCHA and the Google "
    "[Privacy Policy](https://policies.google.com/privacy) and "
    "[Terms of Service](https://policies.google.com/terms) apply."
)


def render_recaptcha(*, action: str = "submit", key: str = "recaptcha") -> str | None:
    site_key = get_recaptcha_site_key()
    if not site_key or site_key == "CHANGE_ME":
        st.warning("reCAPTCHA is not configured.")
        return None

    nonce = st.session_state.get(f"{key}_nonce", 0)
    result = _recaptcha_component(
        data={"site_key": site_key, "action": action, "nonce": nonce},
        key=key,
        default={"token": None},
        on_token_change=lambda: None,
        height=0,
    )
    token = getattr(result, "token", None)

    # reCAPTCHA v3 tokens are single-use. Once a token has been submitted for
    # verification we must not reuse it; return None until a fresh token arrives.
    if token is not None and token == st.session_state.get(f"{key}_used"):
        return None
    return token


def invalidate_recaptcha(key: str, token: str | None) -> None:
    """Mark the current token as consumed and force the widget to fetch a new one."""
    st.session_state[f"{key}_used"] = token
    st.session_state[f"{key}_nonce"] = st.session_state.get(f"{key}_nonce", 0) + 1


def render_recaptcha_disclaimer() -> None:
    st.caption(_RECAPTCHA_DISCLAIMER)
