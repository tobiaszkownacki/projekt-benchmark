import streamlit as st

st.set_page_config(page_title="Benchmark", layout="wide")

from streamlit.navigation.page import StreamlitPage
from auth import repository
from auth.recaptcha_widget import hide_recaptcha_badge
from auth.session import get_current_user, is_logged_in, logout
from auth.join_info_onboarding import render_join_info_onboarding
from auth.login_panel import render_login_panel
from auth.pending_approval import render_pending_approval
from views.admin.admin_panel import render_admin_panel
from views.instructions import render_instructions
from views.leaderboard import render_leaderboard
from views.run_form import render_run_form
from views.run_history import render_run_history
from views.welcome_page import render_welcome_page

def _render_welcome_message(user: repository.User) -> None:
    st.title("Benchmark")
    st.write(f"Welcome, **{user.display_name or user.email}**!")
    st.divider()

def _render_disabled_account() -> None:
    st.error("Your account has been disabled. Please contact an administrator.")
    st.button("Log out", on_click=logout)


def _run_standalone(view) -> None:
    st.navigation([st.Page(view, title="Benchmark")], position="hidden").run()


def _instructions_page() -> StreamlitPage:
    return st.Page(
        render_instructions,
        title="Instructions",
        icon=":material/menu_book:",
        url_path="instructions",
    )

def _render_logged_out_app() -> None:
    def _login_page() -> StreamlitPage:
        return st.Page(
            render_login_panel,
            title="Sign in",
            icon=":material/login:",
            url_path="login",
        )

    login_page = _login_page()

    def _welcome_view() -> None:
        render_welcome_page(login_page)

    welcome_page = st.Page(
        _welcome_view,
        title="Welcome",
        icon=":material/waving_hand:",
        url_path="welcome",
        default=True,
    )

    st.navigation([welcome_page, login_page, _instructions_page()]).run()


def _render_verified_app(user: repository.User) -> None:
    instructions_page = _instructions_page()

    def _home_view() -> None:
        _render_welcome_message(user)
        render_run_form(instructions_page)
        st.divider()
        render_run_history()

    def _leaderboard_view() -> None:
        st.title("Full leaderboard")
        render_leaderboard()

    home_page = st.Page(
        _home_view,
        title="Home",
        icon=":material/home:",
        url_path="home",
        default=True,
    )
    leaderboard_page = st.Page(
        _leaderboard_view,
        title="Full leaderboard",
        icon=":material/leaderboard:",
        url_path="leaderboard",
    )

    with st.sidebar:
        st.markdown(f":material/person: **{user.display_name or user.email}**")
        st.button("Log out", on_click=logout, width="stretch")

    st.navigation([home_page, instructions_page, leaderboard_page]).run()


def _render_admin_app(user: repository.User) -> None:
    def _admin_view() -> None:
        _render_welcome_message(user)
        render_admin_panel()

    def _logs_view() -> None:
        st.title("Logs")
        st.info("TODO: logs")

    admin_page = st.Page(
        _admin_view,
        title="User approval",
        icon=":material/how_to_reg:",
        url_path="admin",
        default=True,
    )
    logs_page = st.Page(
        _logs_view,
        title="Logs",
        icon=":material/article:",
        url_path="logs",
    )

    with st.sidebar:
        st.markdown(f":material/person: **{user.display_name or user.email}**")
        st.button("Log out", on_click=logout, width="stretch")

    st.navigation([admin_page, logs_page]).run()


def _route_authenticated_user(user: repository.User) -> None:
    if not user.is_active:
        _run_standalone(_render_disabled_account)
        return

    # First oauth login users will not have join info, so we need to show the onboarding page
    if not repository.has_join_info(user):
        _run_standalone(lambda: render_join_info_onboarding(user))
        return

    if user.role == "unverified":
        _run_standalone(render_pending_approval)
        return

    if user.role == "admin":
        _render_admin_app(user)
        return

    _render_verified_app(user)


def main() -> None:
    if is_logged_in():
        hide_recaptcha_badge()

    if not is_logged_in():
        _render_logged_out_app()
        return

    user = get_current_user()
    if user is None:
        _render_logged_out_app()
        return

    _route_authenticated_user(user)

if __name__ == "__main__":
    main()
