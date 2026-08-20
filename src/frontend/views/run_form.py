from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any
import streamlit as st

from src.frontend.views.mock_data import DATASETS, OPTIMIZERS
from src.frontend.auth import repository
from src.frontend.core.validator import run_synchronous_validation
from src.task_queue.shared.connectors.rabbitmq_connector import RabbitMQConnector


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

            db_task = repository.create_task(
                queue_name="ATHENA_WORKER_QUEUE",
                executor_name="Athena",
                submitted_by=user.id,
                dataset=dataset.lower(),
                run_name=run_name,
                optimizer_params={
                    "optimizers": optimizers,
                    "custom_codes": custom_optimizers_data
                },
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
