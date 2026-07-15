import streamlit as st
from views.mock_data import RUN_HISTORY

_STATUS_ICON = {
    "completed": ":green[:material/check_circle:]",
    "failed": ":red[:material/cancel:]",
}


def render_run_history() -> None:
    st.subheader("Completed runs history")

    if not RUN_HISTORY:
        st.info("No completed runs.")
        return

    for run in RUN_HISTORY:
        icon = _STATUS_ICON.get(run.status, ":gray[:material/help:]")
        label = f"{icon} {run.run_name}  ·  {run.created_at}"
        with st.expander(label):
            st.info("TODO")
