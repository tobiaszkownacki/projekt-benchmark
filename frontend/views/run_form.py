from typing import Any
import streamlit as st
from views.mock_data import DATASETS, OPTIMIZERS


def render_run_form(instructions_page: Any | None = None) -> None:
    form_col, side_col = st.columns([3, 1])

    with side_col:
        st.markdown("**Need help?**")
        if instructions_page is not None:
            st.page_link(
                instructions_page,
                label="View instructions",
                icon=":material/menu_book:",
                width="stretch",
            )
        else:
            st.caption("Instructions available in the left menu.")

    with form_col:
        with st.form("new_run_form", clear_on_submit=False):
            st.text_input("Run name", placeholder="e.g. lion-imagenet-sweep")
            st.selectbox("Dataset", DATASETS)
            st.multiselect("Optimizers", OPTIMIZERS)
            st.file_uploader(
                "Upload your own optimizers",
                accept_multiple_files=True,
            )
            st.caption("TODO: more run options to be added")

            submitted = st.form_submit_button(
                "Run benchmark", type="primary", width="stretch"
            )

        if submitted:
            st.success("This is a mockup. The UI works correctly.")
