import os
import pika

def get_rabbitmq_connection_params():

    user = os.environ["RABBITMQ_USER"]
    password = os.environ["RABBITMQ_PASSWORD"]
    host = os.environ.get("RABBITMQ_HOST", "rabbitmq")
    port = int(os.environ.get("RABBITMQ_PORT", 5672))

    credentials = pika.PlainCredentials(user, password)
    return pika.ConnectionParameters(
        host=host,
        port=port,
        credentials=credentials,
    )