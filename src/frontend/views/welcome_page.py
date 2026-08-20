from typing import Any
import streamlit as st
from views.leaderboard import render_leaderboard


def render_welcome_page(login_page: Any | None = None) -> None:
    st.title("Benchmark")
    st.subheader("Open platform for comparing optimizers")

    intro, cta = st.columns([3, 1])
    with intro:
        st.markdown(
            "DESC TODO"
        )
    with cta:
        if login_page is not None:
            if st.button(
                "Sign in / Register",
                type="primary",
                width="stretch",
                key="welcome_login_cta",
            ):
                st.switch_page(login_page)

    st.divider()

    st.subheader(":material/trophy: Top results")
    render_leaderboard(limit=5)
    st.caption("Sign in to see the full leaderboard.")
