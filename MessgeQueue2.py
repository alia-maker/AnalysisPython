import json

import message_queue
import pika

# Create you worker method
def my_worker(channel, method, properties, body):
    print( json.loads(body))

if __name__ == '__main__':
    # Instantiate the AMQP adapter with the host configuration
    adapter = message_queue.AMQPAdapter(host='107.23.60.208')
    # Configurate queue
    adapter.configurate_queue(queue='python.publish.test')

    # Instantiate subscriber
    subscriber = message_queue.Subscriber(adapter)
    # Consume message
    subscriber.consume(my_worker)