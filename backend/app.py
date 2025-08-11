#!/usr/bin/env python3
"""
Flask Backend for Brent Oil Price Analysis Dashboard
Task 3: Interactive Dashboard using Flask + React
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

# Setup project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Global data storage
data_cache = {}
analysis_results = {}

def load_data():
    """Load Brent oil price data and analysis results."""
    global data_cache, analysis_results
    
    try:
        from data_processing.loader import DataLoader
        
        # Load data
        loader = DataLoader()
        data = loader.load_existing_data("BrentOilPrices.csv")
        
        if data is None or data.empty:
            data = loader.fetch_brent_oil_data(save_to_file=True)
        
        # Convert to JSON-serializable format
        data_cache = {
            'dates': data['date'].dt.strftime('%Y-%m-%d').tolist(),
            'prices': data['price'].tolist(),
            'date_range': {
                'start': data['date'].min().strftime('%Y-%m-%d'),
                'end': data['date'].max().strftime('%Y-%m-%d')
            },
            'statistics': {
                'min_price': float(data['price'].min()),
                'max_price': float(data['price'].max()),
                'avg_price': float(data['price'].mean()),
                'std_price': float(data['price'].std()),
                'total_records': len(data)
            }
        }
        
        # Load analysis results if available
        results_file = project_root / 'outputs' / 'change_points.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                analysis_results = json.load(f)
        
        return True
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return False

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'message': 'Flask backend is running'})

@app.route('/api/data', methods=['GET'])
def get_data():
    """Get Brent oil price data."""
    if not data_cache:
        if not load_data():
            return jsonify({'error': 'Failed to load data'}), 500
    
    return jsonify(data_cache)

@app.route('/api/data/filtered', methods=['POST'])
def get_filtered_data():
    """Get filtered data based on date range."""
    if not data_cache:
        if not load_data():
            return jsonify({'error': 'Failed to load data'}), 500
    
    filters = request.json
    start_date = filters.get('start_date')
    end_date = filters.get('end_date')
    
    if not start_date or not end_date:
        return jsonify(data_cache)
    
    # Filter data by date range
    dates = pd.to_datetime(data_cache['dates'])
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    mask = (dates >= start_dt) & (dates <= end_dt)
    filtered_dates = [dates[i].strftime('%Y-%m-%d') for i in range(len(dates)) if mask[i]]
    filtered_prices = [data_cache['prices'][i] for i in range(len(dates)) if mask[i]]
    
    filtered_data = {
        'dates': filtered_dates,
        'prices': filtered_prices,
        'date_range': {'start': start_date, 'end': end_date},
        'statistics': {
            'min_price': float(min(filtered_prices)) if filtered_prices else 0,
            'max_price': float(max(filtered_prices)) if filtered_prices else 0,
            'avg_price': float(np.mean(filtered_prices)) if filtered_prices else 0,
            'std_price': float(np.std(filtered_prices)) if filtered_prices else 0,
            'total_records': len(filtered_prices)
        }
    }
    
    return jsonify(filtered_data)

@app.route('/api/changepoints', methods=['GET'])
def get_changepoints():
    """Get detected change points."""
    if not analysis_results:
        if not load_data():
            return jsonify({'error': 'Failed to load analysis results'}), 500
    
    return jsonify(analysis_results)

@app.route('/api/events', methods=['GET'])
def get_events():
    """Get historical events data."""
    # Sample geopolitical events affecting oil prices
    events = [
        {
            'date': '1990-08-02',
            'title': 'Iraq invades Kuwait',
            'description': 'Gulf War begins, causing oil price spike',
            'impact': 'high',
            'type': 'conflict'
        },
        {
            'date': '2001-09-11',
            'title': '9/11 Attacks',
            'description': 'Terrorist attacks in US, market uncertainty',
            'impact': 'high',
            'type': 'terrorism'
        },
        {
            'date': '2008-09-15',
            'title': 'Lehman Brothers Collapse',
            'description': 'Financial crisis, demand destruction',
            'impact': 'high',
            'type': 'economic'
        },
        {
            'date': '2014-11-27',
            'title': 'OPEC Maintains Production',
            'description': 'OPEC decides not to cut production despite low prices',
            'impact': 'high',
            'type': 'policy'
        },
        {
            'date': '2020-03-11',
            'title': 'COVID-19 Pandemic',
            'description': 'Global lockdowns, demand collapse',
            'impact': 'extreme',
            'type': 'pandemic'
        }
    ]
    
    return jsonify({'events': events})

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Get advanced analytics and metrics."""
    if not data_cache:
        if not load_data():
            return jsonify({'error': 'Failed to load data'}), 500
    
    prices = np.array(data_cache['prices'])
    
    # Calculate volatility (rolling 30-day)
    volatility = []
    window = 30
    for i in range(len(prices)):
        start_idx = max(0, i - window + 1)
        window_prices = prices[start_idx:i+1]
        if len(window_prices) > 1:
            vol = np.std(window_prices)
        else:
            vol = 0
        volatility.append(float(vol))
    
    # Calculate returns
    returns = []
    for i in range(1, len(prices)):
        ret = (prices[i] - prices[i-1]) / prices[i-1] * 100
        returns.append(float(ret))
    returns.insert(0, 0)  # First day has no return
    
    analytics = {
        'volatility': volatility,
        'returns': returns,
        'metrics': {
            'avg_volatility': float(np.mean(volatility)),
            'max_volatility': float(np.max(volatility)),
            'avg_return': float(np.mean(returns)),
            'max_return': float(np.max(returns)),
            'min_return': float(np.min(returns))
        }
    }
    
    return jsonify(analytics)

if __name__ == '__main__':
    print("Starting Flask Backend for Brent Oil Analysis Dashboard...")
    print("Loading data...")
    
    if load_data():
        print("✅ Data loaded successfully")
        print(f"   Records: {data_cache.get('statistics', {}).get('total_records', 0)}")
        print(f"   Date range: {data_cache.get('date_range', {})}")
    else:
        print("⚠️  Warning: Could not load data")
    
    print("\n🚀 Starting Flask server...")
    print("Backend will be available at: http://localhost:5000")
    print("API endpoints:")
    print("  GET  /api/health - Health check")
    print("  GET  /api/data - Get all data")
    print("  POST /api/data/filtered - Get filtered data")
    print("  GET  /api/changepoints - Get change points")
    print("  GET  /api/events - Get historical events")
    print("  GET  /api/analytics - Get analytics")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
