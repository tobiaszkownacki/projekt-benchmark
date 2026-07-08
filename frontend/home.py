import streamlit as st

st.set_page_config(page_title="Benchmark", layout="wide")

from auth.constants import ACCOUNT_DISABLED_MESSAGE
from auth.session import get_current_user, is_logged_in, logout
from components.admin_panel import render_admin_panel
from components.login_panel import render_login_panel
from components.pending_approval import render_pending_approval

if not is_logged_in():
    render_login_panel()
    st.stop()

user = get_current_user()
if user is None:
    render_login_panel()
    st.stop()

if not user.is_active:
    st.error(ACCOUNT_DISABLED_MESSAGE)
    st.button("Log out", on_click=logout)
    st.stop()

if user.role == "unverified":
    render_pending_approval()
    st.stop()

st.title("Optimizer Benchmark")
st.write(f"Welcome, **{user.display_name or user.email}**!")

if user.role == "admin":
    render_admin_panel()

st.divider()
st.markdown("Main application content goes here.")

st.button("Log out", on_click=logout)
