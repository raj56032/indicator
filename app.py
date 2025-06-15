from flask import Flask, request, jsonify
from nifty_analyser_fixed import NiftyAnalyzer
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow frontend (React) to call this backend

analyzer = NiftyAnalyzer()

@app.route('/analyse', methods=['GET'])
def analyze():
    ticker = request.args.get('ticker', 'Nifty50')
    interval = request.args.get('interval', '15m')
    period = request.args.get('period', '3d')

    result = analyzer.analyze(ticker, interval, period, plot=False)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
