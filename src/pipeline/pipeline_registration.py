from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class InfrastructureRegistration:
    name: str
    executor_path: str
    # Keyword arguments for the executor: where a centre's profile enters, so
    # that adding a cluster is a registration rather than a subclass.
    executor_config: Mapping[str, Any] = field(default_factory=dict)
    # services
    worker_path: str = "pipeline.generic_services.generic_worker.GenericWorker"
    downloader_path: str = "pipeline.generic_services.generic_downloader.Downloader"
    poller_path: str = "pipeline.generic_services.generic_poller.GenericPoller"
    completion_path: str | None = None  # None if executor is self reporting

    # broker,db
    task_repository_path: str = "pipeline.sql_task_repository.SqlTaskRepository"
    db_connector_path: str = "shared.connectors.postgres.PostGresConnector"
    message_broker_path: str = "shared.connectors.rabbitmq.RabbitMQConnector"


INFRASTRUCTURES: dict[str, InfrastructureRegistration] = {}


def register_new_infrastructure(reg: InfrastructureRegistration) -> None:
    if reg.name in INFRASTRUCTURES:
        raise ValueError(f"Infrastructure '{reg.name}' is already registered")
    INFRASTRUCTURES[reg.name] = reg


def get_infrastructure_registration(name: str) -> InfrastructureRegistration:
    if name not in INFRASTRUCTURES:
        raise ValueError(f"Infrastructure '{name}' is not registered")
    return INFRASTRUCTURES[name]
