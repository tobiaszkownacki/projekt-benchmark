import streamlit as st
from email_validator import validate_email, EmailNotValidError

from auth import repository
from auth.passwords import validate_password_strength
from auth.recaptcha import verify_recaptcha
from auth.session import login_with_email
from auth.join_info_form import render_join_info_inputs, validate_join_info
from auth.recaptcha_widget import (
    invalidate_recaptcha,
    render_recaptcha,
    render_recaptcha_disclaimer,
)

_REGISTER_SUCCESS_KEY = "register_success"
_AUTH_ERROR_KEY = "auth_error"


def _set_auth_error(message: str) -> None:
    st.session_state[_AUTH_ERROR_KEY] = message


def _clear_auth_error() -> None:
    st.session_state.pop(_AUTH_ERROR_KEY, None)


def _render_auth_error() -> None:
    message = st.session_state.get(_AUTH_ERROR_KEY)
    if message:
        st.error(message)


def _validate_email_format(email: str) -> str | None:
    try:
        validate_email(email, check_deliverability=False)
        return None
    except EmailNotValidError as exc:
        return str(exc)


def _validate_register_inputs(email: str, password: str, password2: str) -> str | None:
    email_error = _validate_email_format(email)
    if email_error:
        return email_error

    pwd_error = validate_password_strength(password)
    if pwd_error:
        return pwd_error

    if password != password2:
        return "Passwords do not match."

    return None


def _render_oauth_tab() -> None:
    st.markdown("Sign in with an external account:")
    st.button(
        "Sign in with Google",
        on_click=st.login,
        kwargs={"provider": "google"},
        width="stretch",
    )
    st.button(
        "Sign in with Microsoft",
        on_click=st.login,
        kwargs={"provider": "microsoft"},
        width="stretch",
    )


def _render_login_form() -> None:
    _render_auth_error()
    render_recaptcha_disclaimer()
    
    captcha_token = render_recaptcha(action="login", key="login_recaptcha")

    email = st.text_input("Email", key="login_email")
    password = st.text_input("Password", type="password", key="login_password")

    if st.button("Sign in", width="stretch", key="login_submit"):
        _clear_auth_error()

        if not email or not password:
            st.error("Please enter your email and password.")
            return

        if not captcha_token:
            st.warning("Security check is still loading. Wait a moment and try again.")
            return

        with st.spinner("Signing in..."):
            captcha_ok = verify_recaptcha(captcha_token, action="login")
            invalidate_recaptcha("login_recaptcha", captcha_token)

            if not captcha_ok:
                _set_auth_error("reCAPTCHA verification failed. Please try again.")
                st.rerun()

            ok, message = login_with_email(email, password)

        if ok:
            st.rerun()
        else:
            _set_auth_error(message)
            st.rerun()


def on_go_to_sign_in() -> None:
    st.session_state.pop(_REGISTER_SUCCESS_KEY, None)
    st.session_state["email_auth_mode"] = "Sign in"


def _render_register_form() -> None:
    if st.session_state.get(_REGISTER_SUCCESS_KEY):
        st.success(
            "Account created! You can sign in once an administrator has approved your account."
        )
        st.button("Go to Sign in", width="stretch", key="register_success_continue", on_click = on_go_to_sign_in)
        return

    _render_auth_error()

    render_recaptcha_disclaimer()
    captcha_token = render_recaptcha(action="register", key="register_recaptcha")

    email = st.text_input("Email", key="register_email")
    password = st.text_input("Password", type="password", key="register_password")
    password2 = st.text_input("Confirm password", type="password", key="register_password2")

    join_info, join_mode = render_join_info_inputs("register")

    if st.button("Register", width="stretch", key="register_submit"):
        _clear_auth_error()

        validation_error = _validate_register_inputs(email, password, password2)
        if validation_error:
            st.error(validation_error)
            return

        join_error = validate_join_info(join_info, join_mode)
        if join_error:
            st.error(join_error)
            return

        if not captcha_token:
            st.warning("Security check is still loading. Wait a moment and try again.")
            return

        with st.spinner("Creating your account..."):
            captcha_ok = verify_recaptcha(captcha_token, action="register")
            invalidate_recaptcha("register_recaptcha", captcha_token)

            if not captcha_ok:
                _set_auth_error("reCAPTCHA verification failed. Please try again.")
                st.rerun()

            existing = repository.get_by_email(email)
            if existing:
                if existing.auth_provider != "email":
                    provider = existing.auth_provider.capitalize()
                    _set_auth_error(f"This email is already linked to {provider} login.")
                else:
                    _set_auth_error("An account with this email already exists.")
                st.rerun()

            repository.create_email_user(
                email,
                password,
                associated_organisation=join_info.associated_organisation,
                associated_org_email=join_info.associated_org_email,
                join_reason=join_info.join_reason,
            )

        st.session_state[_REGISTER_SUCCESS_KEY] = True
        st.rerun()


def render_login_panel() -> None:
    st.title("Sign in")

    oauth_tab, email_tab = st.tabs(["OAuth", "Email & password"])

    with oauth_tab:
        _render_oauth_tab()

    with email_tab:
        auth_mode = st.radio(
            "Account action",
            ["Sign in", "Register"],
            horizontal=True,
            label_visibility="collapsed",
            key="email_auth_mode",
            on_change=_clear_auth_error,
        )
        if auth_mode == "Sign in":
            _render_login_form()
        else:
            _render_register_form()
