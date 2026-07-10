import streamlit as st

st.set_page_config(page_title="Benchmark", layout="wide")

from auth import repository
from auth.constants import ACCOUNT_DISABLED_MESSAGE
from auth.session import get_current_user, is_logged_in, logout
from components.admin_panel import render_admin_panel
from components.join_info_onboarding import render_join_info_onboarding
from components.login_panel import render_login_panel
from components.pending_approval import render_pending_approval

def _render_welcome_message(user: repository.User) -> None:
    st.title("Benchmark")
    st.write(f"Welcome, **{user.display_name or user.email}**!")
    st.divider()

def _render_disabled_account() -> None:
    st.error(ACCOUNT_DISABLED_MESSAGE)
    st.button("Log out", on_click=logout)


def _render_main_app(user: repository.User) -> None:
    _render_welcome_message(user)
    st.markdown("Main application content goes here.")

    st.button("Log out", on_click=logout)


def _route_authenticated_user(user: repository.User) -> None:
    if not user.is_active:
        _render_disabled_account()
        return

    # First oauth login users will not have join info, so we need to show the onboarding page
    if not repository.has_join_info(user):
        render_join_info_onboarding(user)
        return

    if user.role == "unverified":
        render_pending_approval()
        return

    if user.role == "admin":
        _render_welcome_message(user)
        render_admin_panel()
        st.button("Log out", on_click=logout)
        return

    _render_main_app(user)


def main() -> None:
    if not is_logged_in():
        render_login_panel()
        return

    user = get_current_user()
    if user is None:
        render_login_panel()
        return

    _route_authenticated_user(user)


main()
