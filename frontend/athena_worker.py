import json
from abc import abstractmethod, ABC

import pika

def setup_rabbitmq():
    

class Worker(ABC):

    @abstractmethod
    def execute(self):
        pass


class SSHWorker(Worker):
    pass


def callback(rabbitmq_channel,method,
             properties,body):

    task_json = json.loads(body.decode("utf-8"))

    worker = SSHWorker()
    try:
        worker.execute(task_json)
        rabbitmq_channel.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as e:
        #TODO update task status to POSTGRES
        print("Failed to execute")
        rabbitmq_channel.basic_nack(delivery_tag=method.delivery_tag
                                    requeue=False)


if __name__ == "__main__":
    conn = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    rabbitmq_channel = conn.channel()

    setup_rabbitmq()


conn = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = conn.channel()

channel.basic_qos(prefetch_count=1)

channel.basic_consume(queue="ATHENA",on_message_callback=callback)

channel.start_consuming()
