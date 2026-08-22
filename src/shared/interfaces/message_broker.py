from abc import abstractmethod

from shared.interfaces.base_connector import BaseConnector


class MessageBrokerConnector(BaseConnector):

    @abstractmethod
    def publish(self):
        pass