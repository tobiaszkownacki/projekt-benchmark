from abc import abstractmethod

from shared.interfaces.base_connector import BaseConnector


class ExecutorConnector(BaseConnector):

    @abstractmethod
    def submit_job(self):
        pass

    @abstractmethod
    def get_job_status(self):
        pass

    @abstractmethod
    def download_results(self):
        pass
