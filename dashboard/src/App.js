// src/App.js
import React, { useEffect, useState } from 'react';
import './App.css';

function App() {
  const [niftyData, setNiftyData] = useState(null);

  useEffect(() => {
    fetch('https://127.0.0.1:5000/analyse')
      .then(response => response.json())
      .then(data => setNiftyData(data))
      .catch(error => console.error('Error fetching data:', error));
  }, []);

  return (
    <div className="App">
      <h1>Nifty Dashboard</h1>
      {niftyData ? (
        <div>
          <p><strong>Trend:</strong> {niftyData.trend}</p>
          <p><strong>Current Price:</strong> {niftyData.current_price}</p>
          <p><strong>RSI:</strong> {niftyData.indicators.RSI}</p>
          <p><strong>MACD:</strong> {niftyData.indicators.MACD}</p>
        </div>
      ) : (
        <p>Loading data...</p>
      )}
    </div>
  );
}

export default App;
