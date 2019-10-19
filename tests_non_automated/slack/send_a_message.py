from experiments.slack_client import SlackNotifier

producer = SlackNotifier.create(channel='batch-jobs', username='Alberto')
producer.send(message='Hello World')
