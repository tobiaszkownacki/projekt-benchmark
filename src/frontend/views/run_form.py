from typing import Any
import httpx
import streamlit as st

from core.config import get_api_base_url
from views.mock_data import DATASETS, OPTIMIZERS
from auth import repository

Executors = ["Athena"]


def render_run_form(instructions_page: Any | None = None, user: repository.User | None = None) -> None:
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
            run_name = st.text_input("Run name", placeholder="e.g. lion-imagenet-sweep")
            dataset = st.selectbox("Dataset", DATASETS)
            optimizers = st.multiselect("Optimizers", OPTIMIZERS)
            uploaded_files = st.file_uploader(
                "Upload your own optimizers (.py)",
                accept_multiple_files=True,
                type=["py"]
            )
            st.caption("TODO: more run options to be added")

            submitted = st.form_submit_button(
                "Run benchmark", type="primary", width="stretch"
            )

        if submitted:
            custom_optimizers_data = {}
            validation_passed = True

            if uploaded_files:
                with st.spinner("Running security checks and validating custom optimizers..."):
                    for uploaded_file in uploaded_files:
                        filename = uploaded_file.name
                        code = uploaded_file.getvalue().decode("utf-8")

                        result = run_synchronous_validation(code, filename)

                        if result["status"] != "success":
                            validation_passed = False
                            st.error(f"Validation failed for `{filename}`. Reason: {result['status'].upper()}")
                            with st.expander("View Validation Logs", expanded=True):
                                st.code(result["logs"], language="text")
                            break
                        else:
                            st.success(f"Validation passed for `{filename}`!")
                            with st.expander("View Validation Logs (Success)", expanded=False):
                                st.code(result["logs"], language="text")
                            custom_optimizers_data[filename] = code

            if not validation_passed:
                st.warning("Benchmark submission aborted due to validation errors. Please fix your code and try again.")
                st.stop()

            if not optimizers and not custom_optimizers_data:
                st.error("You must select at least one built-in optimizer or upload a valid custom optimizer.")
                st.stop()

            all_optimizer_names = optimizers + list(custom_optimizers_data.keys())

            try:
                response = httpx.post(
                    f"{get_api_base_url()}/tasks",
                    json={
                        "dataset": dataset.lower(),
                        "optimizers": all_optimizer_names,
                        "run_name": run_name,
                        "submitted_by": str(user.id),
                    },
                    timeout=10,
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                st.error(f"Failed to submit task: {exc.response.text}")
                return
            except httpx.HTTPError as exc:
                st.error(f"Failed to reach the task API: {exc}")
                return

            st.success("Task submitted successfully")
