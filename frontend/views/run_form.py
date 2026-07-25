import json
from dataclasses import dataclass
from typing import Any
import streamlit as st

from views.mock_data import DATASETS, OPTIMIZERS
from core.config import get_rabbitmq_connection_params
import pika


@dataclass
class TaskDTO:
    run_name: str
    dataset: str
    optimizer: str
    created_at: str
    updated_at: str
    status: str

Executors = ["Athena"]


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

            task_json = {
                "run_name": st.text_input("Run name"),
                "dataset": st.selectbox("Dataset", DATASETS)
            }
            task_json = json.dumps(task_json)
            #TODO diffrent routing keys per executor
            with RabbitMQConnector(exchange="main_exchange") as publisher:
                publisher.publish(task_json)

            st.success("Zadanie wysłane pomyślnie!")



            st.success("This is a mockup. The UI works correctly.")

class RabbitMQConnector:

    def __init__(self, exchange: str, routing_key: str) -> None:
        self.exchange = exchange
        self.routing_key = routing_key
        self.connection = None

    def __enter__(self):
        credentials = get_rabbitmq_connection_params()
        self.connection = pika.BlockingConnection(credentials)
        self.channel = self.connection.channel()
        self.channel.exchange_declare(exchange=self.exchange, exchange_type="direct", durable=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        if self.connection and not self.connection.is_closed:
            self.connection.close()
        if exc_type:
            st.error(f"ERROR RABBITMQ: {exc_val}")

    def publish(self,task_dict: TaskDTO) -> None:
        payload = json.dumps(task_dict)
        self.channel.basic_publish(
            exchange=self.exchange,
            routing_key=self.routing_key,
            body=payload,
            properties=pika.BasicProperties(
                delivery_mode=pika.DeliveryMode.Persistent,
                content_type='application/json'
            )
        )
