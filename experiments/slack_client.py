from slack import WebClient
import os

from experiments.logger import logger

SLACK_TOKEN = 'SLACK_API_TOKEN'


class Notifier:
    def send(self, message):
        pass


class SlackNotifier(Notifier):
    @staticmethod
    def create(channel):
        # get it here: https://api.slack.com/apps/APKLV82ER/oauth?
        slack_token = os.getenv(SLACK_TOKEN)
        return SlackNotifier(slack_token=slack_token,
                             channel=channel)

    def __init__(self, slack_token, channel):
        self.slack_token = slack_token
        self.channel = channel

    def send(self, message):
        try:
            slack_client = WebClient(token=self.slack_token)
            result = slack_client.chat_postMessage(
                channel=self.channel,
                text=message
            )
            if not result['ok']:
                # TODO log error
                print(f'        SlackProducer failed to send message: {message}')
        except Exception as e:
            logger.exception(e)

        return result
