import streamlit as st

from views.mock_data import LEADERBOARD


def render_leaderboard(limit: int | None = None) -> None:
    entries = LEADERBOARD if limit is None else LEADERBOARD[:limit]

    rows = [
        {
            "#": entry.rank,
            "Optimizer": entry.optimizer,
            "Dataset": entry.dataset,
            "Score": entry.score,
            "Date": entry.submitted_at,
        }
        for entry in entries
    ]

    st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score",
                min_value=0,
                max_value=100,
                format="%.1f",
            ),
        },
    )
