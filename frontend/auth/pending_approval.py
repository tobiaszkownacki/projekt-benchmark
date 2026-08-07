import streamlit as st

from auth.session import logout


def render_pending_approval() -> None:
    st.title("Account pending approval")
    st.info(
        "Your account is awaiting approval by an administrator. "
        "Refresh the page once you have been granted access."
    )
    st.button("Log out", on_click=logout)
