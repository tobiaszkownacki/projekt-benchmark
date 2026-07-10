import streamlit as st
from email_validator import validate_email, EmailNotValidError

from auth import repository
from auth.passwords import validate_password_strength
from auth.recaptcha import verify_recaptcha
from auth.session import login_with_email
from components.join_info_form import render_join_info_inputs, validate_join_info
from components.recaptcha_widget import render_recaptcha, render_recaptcha_disclaimer

_REGISTER_SUCCESS_KEY = "register_success"


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
        use_container_width=True,
    )
    st.button(
        "Sign in with Microsoft",
        on_click=st.login,
        kwargs={"provider": "microsoft"},
        use_container_width=True,
    )


def _render_login_form() -> None:
    with st.form("email_login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in", use_container_width=True)

    if submitted:
        if not email or not password:
            st.error("Please enter your email and password.")
            return
        ok, message = login_with_email(email, password)
        if ok:
            st.rerun()
        else:
            st.error(message)


def on_go_to_sign_in() -> None:
    st.session_state.pop(_REGISTER_SUCCESS_KEY, None)
    st.session_state["email_auth_mode"] = "Sign in"


def _render_register_form() -> None:
    if st.session_state.get(_REGISTER_SUCCESS_KEY):
        st.success(
            "Account created! You can sign in once an administrator has approved your account."
        )
        st.button("Go to Sign in", use_container_width=True, key="register_success_continue", on_click = on_go_to_sign_in)
        return

    render_recaptcha_disclaimer()
    captcha_token = render_recaptcha(action="register", key="register_recaptcha")

    email = st.text_input("Email", key="register_email")
    password = st.text_input("Password", type="password", key="register_password")
    password2 = st.text_input("Confirm password", type="password", key="register_password2")

    join_info, join_mode = render_join_info_inputs("register")

    if st.button("Register", use_container_width=True, key="register_submit"):
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
            if not verify_recaptcha(captcha_token, action="register"):
                st.error("reCAPTCHA verification failed. Please try again.")
                return

            existing = repository.get_by_email(email)
            if existing:
                if existing.auth_provider != "email":
                    provider = existing.auth_provider.capitalize()
                    st.error(f"This email is already linked to {provider} login.")
                else:
                    st.error("An account with this email already exists.")
                return

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
        )
        if auth_mode == "Sign in":
            _render_login_form()
        else:
            _render_register_form()
