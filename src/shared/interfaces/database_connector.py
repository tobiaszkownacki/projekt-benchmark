from abc import abstractmethod

from shared.interfaces.base_connector import BaseConnector


class DatabaseConnector(BaseConnector):

    @abstractmethod
    def execute(self):
        pass