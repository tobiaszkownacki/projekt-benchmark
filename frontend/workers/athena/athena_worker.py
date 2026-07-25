
import json
import os
from abc import abstractmethod, ABC

import pika

from frontend.workers.abstract_worker import Worker


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
        print(e)
        print("Failed to execute")
        rabbitmq_channel.basic_nack(delivery_tag=method.delivery_tag,
                                    requeue=False)
    finally:
        rabbitmq_channel.basic_nack(delivery_tag=method.delivery_tag,
                                    requeue=False)


if __name__ == "__main__":
    conn = pika.BlockingConnection(get_rabbitmq_connection_params())
    channel = conn.channel()

    channel.basic_qos(prefetch_count=1)

    channel.basic_consume(queue="ATHENA_QUEUE",on_message_callback=callback)

    channel.start_consuming()
