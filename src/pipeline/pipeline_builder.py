import importlib
import os

import pipeline.registered_pipelines
from pipeline.pipeline_registration import get_infrastructure_registration
from pipeline.task_repository import TaskRepository
from shared.queue_topology import QueueTopology


def _load_from_class_path(path: str):
    module_path, _, attr = path.rpartition(".")
    return getattr(importlib.import_module(module_path), attr)


class PipelineBuilder:

    def __init__(self):
        try:
            executor_name = os.environ["EXECUTOR"]
        except KeyError:
            raise Exception("EXECUTOR environment variable not set")

        self.registration = get_infrastructure_registration(executor_name)
        self.topology = QueueTopology(executor_name)

    def build(self, service2run: str):
        match service2run:
            case "worker":
                return self._get_worker()
            case "downloader":
                return self._get_downloader()
            case "poller":
                return self._get_poller()
            case _:
                raise ValueError(f"Unknown service to run {service2run}")

    def _build_task_repo(self) -> TaskRepository:
        repo_cls = _load_from_class_path(self.registration.task_repository_path)
        db_cls = _load_from_class_path(self.registration.db_connector_path)
        return repo_cls(db_cls)

    def _build_broker(self):
        return _load_from_class_path(self.registration.message_broker_path)

    def _build_executor(self):
        executor_cls = _load_from_class_path(self.registration.executor_path)
        return executor_cls()

    def _get_worker(self):
        worker_cls = _load_from_class_path(self.registration.worker_path)
        return worker_cls(self._build_executor(), self.topology, self._build_task_repo(), self._build_broker())

    def _get_downloader(self):
        downloader_cls = _load_from_class_path(self.registration.downloader_path)
        return downloader_cls(self._build_executor(), self.topology, self._build_task_repo(), self._build_broker())

    def _get_poller(self):
        if self.registration.completion_path is None:
            raise ValueError(
                f"Executor '{self.registration.name}' has no completion_path; it self-reports completion, no poller"
            )
        poller_cls = _load_from_class_path(self.registration.poller_path)
        completion_cls = _load_from_class_path(self.registration.completion_path)
        completion_rule = completion_cls(self._build_executor(), self._build_task_repo())
        interval_s = int(os.environ.get("POLL_INTERVAL_S", "60"))
        return poller_cls(completion_rule, self.topology, self._build_broker(), interval_s)
