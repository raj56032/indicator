import requests

data = {
    "message": "🚨 Test Signal:\nBuy Nifty 22500 CE\nEntry: ₹90\nSL: ₹65\nTarget: ₹125"
}

response = requests.post("http://127.0.0.1:5000/webhook", json=data)
print(response.text)
