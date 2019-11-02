from experiments.slack_client import SlackNotifier

producer = SlackNotifier.create(channel='batch-jobs')
producer.send(message='Hello World')
