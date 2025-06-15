from flask import Flask, request
import requests

app = Flask(__name__)

# Your Telegram bot token and chat ID
BOT_TOKEN = "7897172055:AAFxIazzKY2clKi4It_p1_Nba7Wo0_HePJA"
CHAT_ID = "1196414502"

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    requests.post(url, json=payload)

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    message = f"📢 *Trade Signal*\n\n{data.get('message', 'No message sent')}"
    send_telegram_message(message)
    return {"status": "ok"}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
