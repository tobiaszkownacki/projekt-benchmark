from dataclasses import dataclass


@dataclass(frozen=True)
class InfrastructureRegistration:
    name: str
    executor_path: str
    #services
    worker_path: str = "pipeline.generic_services.generic_worker.GenericWorker"
    downloader_path: str = "pipeline.generic_services.generic_downloader.Downloader"
    poller_path: str = "pipeline.generic_services.generic_poller.GenericPoller"
    completion_path: str | None = None  # None if executor is self reporting

    # broker,db
    task_repository_path: str = "pipeline.sql_task_repository.SqlTaskRepository"
    db_connector_path: str = "shared.connectors.postgres_connector.PostGresConnector"
    message_broker_path: str = "shared.connectors.rabbitmq_connector.RabbitMQConnector"


INFRASTRUCTURES: dict[str, InfrastructureRegistration] = {}


def register_new_infrastructure(reg: InfrastructureRegistration) -> None:
    if reg.name in INFRASTRUCTURES:
        raise ValueError(f"Infrastructure '{reg.name}' is already registered")
    INFRASTRUCTURES[reg.name] = reg


def get_infrastructure_registration(name: str) -> InfrastructureRegistration:
    if name not in INFRASTRUCTURES:
        raise ValueError(f"Infrastructure '{name}' is not registered")
    return INFRASTRUCTURES[name]
