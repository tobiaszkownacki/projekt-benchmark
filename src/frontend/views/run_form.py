from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any
import streamlit as st

from views.mock_data import DATASETS, OPTIMIZERS
from connectors.rabbitmq_connector import RabbitMQConnector
from auth import repository


@dataclass
class TaskDTO:
    task_id: str
    run_name: str
    dataset: str
    optimizer: str
    submitted_by: str
    created_at: str
    updated_at: str
    status: str

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
            st.file_uploader(
                "Upload your own optimizers",
                accept_multiple_files=True,
            )
            st.caption("TODO: more run options to be added")

            submitted = st.form_submit_button(
                "Run benchmark", type="primary", width="stretch"
            )

        if submitted:

            db_task = repository.create_task(
                queue_name="ATHENA_WORKER_QUEUE",
                executor_name="Athena",
                submitted_by=user.id,
                dataset=dataset.lower(),
                run_name=run_name,
                optimizer_params={"optimizers": optimizers},
            )

            now = datetime.now(timezone.utc).isoformat()
            task = TaskDTO(
                task_id=db_task.task_id,
                run_name=run_name,
                dataset=dataset.lower(),
                optimizer=",".join(name.lower() for name in optimizers),
                submitted_by=str(user.id),
                created_at=now,
                updated_at=now,
                status="pending",
            )
            #TODO diffrent routing keys per executor
            try:
                rmq_secrets = st.secrets["rabbitmq"]
                with RabbitMQConnector(
                    exchange="main-exchange",
                    routing_key="ATHENA_WORKER_QUEUE",
                    user=rmq_secrets["user"],
                    password=rmq_secrets["password"],
                    host=rmq_secrets["host"],
                    port=rmq_secrets["port"],
                ) as publisher:
                    publisher.publish(asdict(task))
            except Exception as exc:
                st.error(f"ERROR RABBITMQ: {exc}")
                return

            st.success("Task send successfully")
            st.success("This is a mockup. The UI works correctly.")
