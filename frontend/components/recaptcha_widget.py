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

    return () => {
        document.querySelectorAll(".grecaptcha-badge").forEach((el) => el.remove());
    };
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

    result = _recaptcha_component(
        data={"site_key": site_key, "action": action},
        key=key,
        default={"token": None},
        on_token_change=lambda: None,
        height=0,
    )
    return getattr(result, "token", None)


def render_recaptcha_disclaimer() -> None:
    st.caption(_RECAPTCHA_DISCLAIMER)
