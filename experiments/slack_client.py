from slack import WebClient
import os
SLACK_TOKEN = 'SLACK_API_TOKEN'


class SlackNotifier:
    @staticmethod
    def create(channel):
        # get it here: https://api.slack.com/apps/APKLV82ER/oauth?
        slack_token = os.getenv(SLACK_TOKEN)
        return SlackNotifier(slack_client=WebClient(token=slack_token),
                             channel=channel)

    def __init__(self, slack_client, channel):
        self.slack_client = slack_client
        self.channel = channel

    def send(self, message):

        result = self.slack_client.chat_postMessage(
            channel=self.channel,
            text=message
        )
        if not result['ok']:
            # TODO log error
            print(f'        SlackProducer failed to send message: {message}')

        return result
