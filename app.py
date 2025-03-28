
# from datetime import datetime  # Import only datetime, not the entire module
# from flask import Flask, render_template, request
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# import datetime
# import json
# import yfinance as yf
# from functools import lru_cache
# import time
# import tensorflow as tf
# from datetime import datetime as dt
# from flask import Flask, render_template

# app = Flask(__name__)

# @app.context_processor
# def inject_datetime():
#     return {
#         'datetime': dt,
#         'now': dt.now(),
#         'current_year': dt.now().strftime('%Y')
#     }

# # Indian stock options (NSE tickers with .NS suffix)
# STOCK_OPTIONS = {
#     "RELIANCE.NS": "Reliance Industries Ltd.",
#     "TCS.NS": "Tata Consultancy Services Ltd.",
#     "HDFCBANK.NS": "HDFC Bank Ltd.",
#     "INFY.NS": "Infosys Ltd.",
#     "SBIN.NS": "State Bank of India",
#     # ... (rest of the STOCK_OPTIONS dictionary remains the same)
#     "TORNTPHARM.NS": "Torrent Pharmaceuticals Ltd."
# }

# # Cache with 15-minute expiration
# def get_ttl_hash(seconds=900):
#     return round(time.time() / seconds)

# @lru_cache(maxsize=32)
# def cached_stock_data(ticker, ttl_hash=None):
#     del ttl_hash  # To make time-based cache work
#     return get_stock_data(ticker)

# def get_stock_data(ticker):
#     """Fetch stock data using yfinance with multiple fallback attempts"""
#     attempts = [
#         {'period': '1mo', 'interval': '1d'},
#         {'period': '3mo', 'interval': '1d'},
#         {'period': '7d', 'interval': '1h'},
#         {'period': '5d', 'interval': '15m'}
#     ]
    
#     for attempt in attempts:
#         try:
#             data = yf.Ticker(ticker).history(
#                 period=attempt['period'],
#                 interval=attempt['interval']
#             )['Close']
#             if len(data) >= 60:
#                 return data
#         except Exception as e:
#             print(f"Attempt failed ({attempt}): {e}")
    
#     return pd.Series()

# def get_stock_info(ticker):
#     """Get sector, previous close price, and company website"""
#     try:
#         stock = yf.Ticker(ticker)
#         info = stock.info
#         return {
#             'sector': info.get('sector', 'N/A'),
#             'current_price': info.get('previousClose', 0),
#             'website': info.get('website', '#'),
#             'logo_url': info.get('logo_url', '')
#         }
#     except:
#         return {'sector': 'N/A', 'current_price': 0, 'website': '#', 'logo_url': ''}

# def prepare_data(data, look_back=60):
#     """Prepare data for LSTM model"""
#     data_values = data.values.astype('float32')
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(data_values.reshape(-1, 1))
    
#     X, y = [], []
#     for i in range(look_back, len(scaled_data)):
#         X.append(scaled_data[i-look_back:i, 0])
#         y.append(scaled_data[i, 0])
    
#     X, y = np.array(X), np.array(y)
#     X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
#     train_size = int(len(X) * 0.8)
#     X_train, X_test = X[:train_size], X[train_size:]
#     y_train, y_test = y[:train_size], y[train_size:]
    
#     return X_train, y_train, X_test, y_test, scaler

# def create_lstm_model(input_shape):
#     """Create LSTM model with improved architecture"""
#     model = Sequential([
#         LSTM(units=100, return_sequences=True, input_shape=input_shape),
#         Dropout(0.3),
#         LSTM(units=100, return_sequences=True),
#         Dropout(0.3),
#         LSTM(units=50),
#         Dropout(0.3),
#         Dense(units=25, activation='relu'),
#         Dense(units=1)
#     ])
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     return model

# def train_model(X_train, y_train):
#     """Train model with early stopping"""
#     model = create_lstm_model((X_train.shape[1], 1))
    
#     early_stop = tf.keras.callbacks.EarlyStopping(
#         monitor='loss', 
#         patience=5,
#         restore_best_weights=True
#     )
    
#     model.fit(
#         X_train, 
#         y_train, 
#         epochs=100, 
#         batch_size=16, 
#         verbose=0,
#         callbacks=[early_stop]
#     )
    
#     return model

# def get_live_price(ticker):
#     """Get live stock price"""
#     try:
#         stock = yf.Ticker(ticker)
#         live_price = stock.history(period='1d')['Close'].iloc[-1]
#         return float(live_price)
#     except Exception as e:
#         print(f"Error getting live price: {e}")
#         return None

# def predict_stock_movement(ticker):
#     """Main prediction function with error handling and live price"""
#     try:
#         # Get live price first
#         live_price = get_live_price(ticker)
        
#         stock_prices = cached_stock_data(ticker, get_ttl_hash())
        
#         if stock_prices.empty:
#             raise ValueError("No data available for this stock")
#         if len(stock_prices) < 60:
#             raise ValueError(f"Only {len(stock_prices)} data points available (need 60)")
        
#         # Prepare data and train model
#         X_train, y_train, _, _, scaler = prepare_data(stock_prices)
#         model = create_lstm_model((X_train.shape[1], 1))
#         model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
        
#         # Prepare last 60 data points for prediction
#         last_60 = stock_prices[-60:].values.astype('float32')
#         last_60_scaled = scaler.transform(last_60.reshape(-1, 1))
#         X_pred = np.array([last_60_scaled.flatten()])
#         X_pred = np.reshape(X_pred, (X_pred.shape[0], X_pred.shape[1], 1))
        
#         # Generate predictions
#         predictions = []
#         for _ in range(100):
#             pred = model.predict(X_pred, verbose=0)
#             predictions.append(scaler.inverse_transform(pred)[0][0])
        
#         predicted_value = np.mean(predictions)
#         last_price = stock_prices.iloc[-1]
        
#         movement = "Up" if predicted_value > last_price else "Down"
#         confidence = abs((predicted_value - last_price) / last_price) * 100
        
#         # Ensure predicted_value is a float
#         predicted_value = float(predicted_value)
        
#         # Prepare chart data
#         dates = stock_prices.index[-60:].strftime('%Y-%m-%d %H:%M').tolist()
#         prices = stock_prices[-60:].values.tolist()
#         chart_data = (dates, prices)
        
#         # Get stock info
#         stock_info = get_stock_info(ticker)
        
#         return (movement, confidence, predicted_value, last_price,
#                 chart_data, stock_info, live_price)
    
#     except Exception as e:
#         print(f"Prediction error: {e}")
#         stock_info = get_stock_info(ticker)
#         return (f"Error: {str(e)}", 0.0, stock_info['current_price'], 
#                 stock_info['current_price'], ([], []), stock_info, None)

# @app.route('/', methods=['GET', 'POST'])
# def home():
#     if request.method == 'POST':
#         ticker = request.form['ticker']
#         stock_name = STOCK_OPTIONS.get(ticker, ticker)
        
#         result = predict_stock_movement(ticker)
#         (movement, confidence, predicted_value, last_price,
#          chart_data, stock_info, live_price) = result
        
#         return render_template('result.html',
#                             ticker=stock_name,
#                             movement=movement,
#                             confidence=f"{confidence:.2f}" if isinstance(confidence, float) else "N/A",
#                             predicted_value=f"{predicted_value:.2f}" if isinstance(predicted_value, float) else "N/A",
#                             last_price=f"{last_price:.2f}" if isinstance(last_price, float) else "N/A",
#                             live_price=f"{live_price:.2f}" if live_price else "N/A",
#                             chart_data=json.dumps(chart_data),
#                             sector=stock_info['sector'],
#                             current_price=f"{stock_info['current_price']:.2f}" if isinstance(stock_info['current_price'], float) else "N/A",
#                             website=stock_info['website'],
#                             logo_url=stock_info['logo_url'],
#                             timestamp=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
#     return render_template('index.html', stock_options=STOCK_OPTIONS)

# if __name__ == '__main__':
#     app.run(debug=True)


from datetime import datetime  # Import only datetime, not the entire module
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import datetime
import json
import yfinance as yf
from functools import lru_cache
import time
import tensorflow as tf
from datetime import datetime as dt
from flask import Flask, render_template

app = Flask(__name__)

@app.context_processor
def inject_datetime():
    return {
        'datetime': dt,
        'now': dt.now(),
        'current_year': dt.now().strftime('%Y')
    }

# Indian stock options (NSE tickers with .NS suffix)
STOCK_OPTIONS = {"RELIANCE.NS": "Reliance Industries Ltd.",
    "TCS.NS": "Tata Consultancy Services Ltd.",
    "HDFCBANK.NS": "HDFC Bank Ltd.",
    "INFY.NS": "Infosys Ltd.",
    "SBIN.NS": "State Bank of India",
    "ICICIBANK.NS": "ICICI Bank Ltd.",
    "HINDUNILVR.NS": "Hindustan Unilever Ltd.",
    "BHARTIARTL.NS": "Bharti Airtel Ltd.",
    "LT.NS": "Larsen & Toubro Ltd.",
    "BAJFINANCE.NS": "Bajaj Finance Ltd.",
    "AXISBANK.NS": "Axis Bank Ltd.",
    "MARUTI.NS": "Maruti Suzuki India Ltd.",
    "KOTAKBANK.NS": "Kotak Mahindra Bank Ltd.",
    "ITC.NS": "ITC Ltd.",
    "ASIANPAINT.NS": "Asian Paints Ltd.",
    "HCLTECH.NS": "HCL Technologies Ltd.",
    "TITAN.NS": "Titan Company Ltd.",
    "SUNPHARMA.NS": "Sun Pharmaceutical Industries Ltd.",
    "M&M.NS": "Mahindra & Mahindra Ltd.",
    "ADANIENT.NS": "Adani Enterprises Ltd.",
    "ULTRACEMCO.NS": "UltraTech Cement Ltd.",
    "NESTLEIND.NS": "Nestlé India Ltd.",
    "WIPRO.NS": "Wipro Ltd.",
    "POWERGRID.NS": "Power Grid Corporation of India Ltd.",
    "ONGC.NS": "Oil and Natural Gas Corporation Ltd.",
    "NTPC.NS": "NTPC Ltd.",
    "BAJAJ-AUTO.NS": "Bajaj Auto Ltd.",
    "TECHM.NS": "Tech Mahindra Ltd.",
    "COALINDIA.NS": "Coal India Ltd.",
    "DRREDDY.NS": "Dr. Reddy's Laboratories Ltd.",
    "CIPLA.NS": "Cipla Ltd.",
    "EICHERMOT.NS": "Eicher Motors Ltd.",
    "GRASIM.NS": "Grasim Industries Ltd.",
    "HEROMOTOCO.NS": "Hero MotoCorp Ltd.",
    "BRITANNIA.NS": "Britannia Industries Ltd.",
    "ADANIPORTS.NS": "Adani Ports and Special Economic Zone Ltd.",
    "SHREECEM.NS": "Shree Cement Ltd.",
    "DIVISLAB.NS": "Divi's Laboratories Ltd.",
    "JSWSTEEL.NS": "JSW Steel Ltd.",
    "TATASTEEL.NS": "Tata Steel Ltd.",
    "HINDALCO.NS": "Hindalco Industries Ltd.",
    "BPCL.NS": "Bharat Petroleum Corporation Ltd.",
    "IOC.NS": "Indian Oil Corporation Ltd.",
    "HINDPETRO.NS": "Hindustan Petroleum Corporation Ltd.",
    "ZOMATO.NS": "Zomato Ltd.",
    "DMART.NS": "Avenue Supermarts Ltd.",
    "PIDILITIND.NS": "Pidilite Industries Ltd.",
    "GODREJCP.NS": "Godrej Consumer Products Ltd.",
    "DABUR.NS": "Dabur India Ltd.",
    "INDUSINDBK.NS": "IndusInd Bank Ltd.",
    "SBILIFE.NS": "SBI Life Insurance Company Ltd.",
    "HDFCLIFE.NS": "HDFC Life Insurance Company Ltd.",
    "BAJAJFINSV.NS": "Bajaj Finserv Ltd.",
    "AMBUJACEM.NS": "Ambuja Cements Ltd.",
    "ACC.NS": "ACC Ltd.",
    "TATAMOTORS.NS": "Tata Motors Ltd.",
    "SIEMENS.NS": "Siemens Ltd.",
    "HAVELLS.NS": "Havells India Ltd.",
    "POLYCAB.NS": "Polycab India Ltd.",
    "TRENT.NS": "Trent Ltd.",
    "VEDL.NS": "Vedanta Ltd.",
    "JINDALSTEL.NS": "Jindal Steel & Power Ltd.",
    "DLF.NS": "DLF Ltd.",
    "INDIGO.NS": "InterGlobe Aviation Ltd.",
    "BANKBARODA.NS": "Bank of Baroda",
    "PNB.NS": "Punjab National Bank",
    "CANBK.NS": "Canara Bank",
    "GAIL.NS": "GAIL (India) Ltd.",
    "APOLLOHOSP.NS": "Apollo Hospitals Enterprise Ltd.",
    "LUPIN.NS": "Lupin Ltd.",
    "AUROPHARMA.NS": "Aurobindo Pharma Ltd.",
    "MRF.NS": "MRF Ltd.",
    "ASHOKLEY.NS": "Ashok Leyland Ltd.",
    "LICHSGFIN.NS": "LIC Housing Finance Ltd.",
    "CONCOR.NS": "Container Corporation of India Ltd.",
    "BEL.NS": "Bharat Electronics Ltd.",
    "HAL.NS": "Hindustan Aeronautics Ltd.",
    "NAUKRI.NS": "Info Edge (India) Ltd.",
    "PERSISTENT.NS": "Persistent Systems Ltd.",
    "LTTS.NS": "L&T Technology Services Ltd.",
    "MPHASIS.NS": "MphasiS Ltd.",
    "COFORGE.NS": "Coforge Ltd.",
    "TATAPOWER.NS": "Tata Power Company Ltd.",
    "ABCAPITAL.NS": "Aditya Birla Capital Ltd.",
    "YESBANK.NS": "Yes Bank Ltd.",
    "IDFCFIRSTB.NS": "IDFC First Bank Ltd.",
    "BANDHANBNK.NS": "Bandhan Bank Ltd.",
    "UPL.NS": "UPL Ltd.",
    "BERGEPAINT.NS": "Berger Paints India Ltd.",
    "COLPAL.NS": "Colgate-Palmolive (India) Ltd.",
    "MCDOWELL-N.NS": "United Spirits Ltd.",
    "BALKRISIND.NS": "Balkrishna Industries Ltd.",
    "TVSMOTOR.NS": "TVS Motor Company Ltd.",
    "EXIDEIND.NS": "Exide Industries Ltd.",
    "MOTHERSON.NS": "Motherson Sumi Systems Ltd.",
    "NMDC.NS": "NMDC Ltd.",
    "SAIL.NS": "Steel Authority of India Ltd.",
    "IRCTC.NS": "Indian Railway Catering and Tourism Corporation Ltd.",
    "PAGEIND.NS": "Page Industries Ltd.",
    "TORNTPHARM.NS": "Torrent Pharmaceuticals Ltd."
}
# Cache with 15-minute expiration
def get_ttl_hash(seconds=900):
    return round(time.time() / seconds)

@lru_cache(maxsize=32)
def cached_stock_data(ticker, ttl_hash=None):
    del ttl_hash  # To make time-based cache work
    return get_stock_data(ticker)

def get_stock_data(ticker):
    """Fetch stock data using yfinance with multiple fallback attempts"""
    attempts = [
        {'period': '1mo', 'interval': '1d'},
        {'period': '3mo', 'interval': '1d'},
        {'period': '7d', 'interval': '1h'},
        {'period': '5d', 'interval': '15m'}
    ]
    
    for attempt in attempts:
        try:
            data = yf.Ticker(ticker).history(
                period=attempt['period'],
                interval=attempt['interval']
            )['Close']
            if len(data) >= 60:
                return data
        except Exception as e:
            print(f"Attempt failed ({attempt}): {e}")
    
    return pd.Series()

def get_stock_info(ticker):
    """Get sector, previous close price, and company website"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'sector': info.get('sector', 'N/A'),
            'current_price': info.get('previousClose', 0),
            'website': info.get('website', '#'),
            'logo_url': info.get('logo_url', '')
        }
    except:
        return {'sector': 'N/A', 'current_price': 0, 'website': '#', 'logo_url': ''}

def advanced_stock_analysis(ticker):
    """Comprehensive stock analysis with multiple indicators"""
    try:
        # Fetch historical stock data
        stock = yf.Ticker(ticker)
        hist_data = stock.history(period='1y')
        
        # Basic Price Analysis
        current_price = hist_data['Close'][-1]
        avg_price_52_weeks_high = hist_data['High'].max()
        avg_price_52_weeks_low = hist_data['Low'].min()
        price_distance_from_52_week_high = ((current_price - avg_price_52_weeks_low) / 
                                            (avg_price_52_weeks_high - avg_price_52_weeks_low)) * 100
        
        # Volume Analysis
        avg_volume = hist_data['Volume'].mean()
        current_volume = hist_data['Volume'][-1]
        volume_change = ((current_volume - avg_volume) / avg_volume) * 100
        
        # Technical Indicators
        # Simple Moving Averages
        ma_20 = hist_data['Close'].rolling(window=20).mean()[-1]
        ma_50 = hist_data['Close'].rolling(window=50).mean()[-1]
        ma_200 = hist_data['Close'].rolling(window=200).mean()[-1]
        
        # Relative Strength Index (RSI)
        def calculate_rsi(data, periods=14):
            delta = data.diff()
            dUp, dDown = delta.copy(), delta.copy()
            dUp[dUp < 0] = 0
            dDown[dDown > 0] = 0
            
            RollUp = dUp.rolling(window=periods).mean()
            RollDown = dDown.abs().rolling(window=periods).mean()
            
            RS = RollUp / RollDown
            RSI = 100.0 - (100.0 / (1.0 + RS))
            return RSI
        
        rsi = calculate_rsi(hist_data['Close'])[-1]
        
        # Decision Logic
        decision_score = 0
        decision_reasons = []
        
        # Price Trend Analysis
        if current_price > ma_20 and current_price > ma_50 and current_price > ma_200:
            decision_score += 2
            decision_reasons.append("Strong Upward Price Trend")
        elif current_price < ma_20 and current_price < ma_50 and current_price < ma_200:
            decision_score -= 2
            decision_reasons.append("Weak Downward Price Trend")
        
        # Volume Analysis
        if current_volume > avg_volume * 1.5:
            decision_score += 1
            decision_reasons.append("High Trading Volume")
        
        # RSI Analysis
        if rsi < 30:
            decision_score += 2
            decision_reasons.append("Oversold Condition (Potential Buy Signal)")
        elif rsi > 70:
            decision_score -= 2
            decision_reasons.append("Overbought Condition (Potential Sell Signal)")
        
        # Price Positioning
        if price_distance_from_52_week_high < 20:
            decision_score += 1
            decision_reasons.append("Close to 52-Week High")
        elif price_distance_from_52_week_high > 80:
            decision_score -= 1
            decision_reasons.append("Far from 52-Week High")
        
        # Final Recommendation
        if decision_score > 2:
            recommendation = "Strong Buy"
            recommendation_color = "green"
        elif decision_score > 0:
            recommendation = "Moderate Buy"
            recommendation_color = "lime"
        elif decision_score == 0:
            recommendation = "Neutral"
            recommendation_color = "orange"
        elif decision_score > -2:
            recommendation = "Moderate Sell"
            recommendation_color = "red"
        else:
            recommendation = "Strong Sell"
            recommendation_color = "darkred"
        
        # Financial Fundamentals
        try:
            financials = {
                'Market Cap': stock.info.get('marketCap', 'N/A'),
                'P/E Ratio': stock.info.get('trailingPE', 'N/A'),
                'Dividend Yield': stock.info.get('dividendYield', 'N/A'),
                'EPS': stock.info.get('trailingEps', 'N/A'),
                'Beta': stock.info.get('beta', 'N/A')
            }
        except:
            financials = {}
        
        return {
            'current_price': current_price,
            '52_week_high': avg_price_52_weeks_high,
            '52_week_low': avg_price_52_weeks_low,
            'price_distance_from_52week_high': price_distance_from_52_week_high,
            'ma_20': ma_20,
            'ma_50': ma_50,
            'ma_200': ma_200,
            'rsi': rsi,
            'volume_change': volume_change,
            'recommendation': recommendation,
            'recommendation_color': recommendation_color,
            'decision_reasons': decision_reasons,
            'financials': financials
        }
    
    except Exception as e:
        print(f"Advanced analysis error: {e}")
        return None

def prepare_data(data, look_back=60):
    """Prepare data for LSTM model"""
    data_values = data.values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, y_train, X_test, y_test, scaler

def create_lstm_model(input_shape):
    """Create LSTM model with improved architecture"""
    model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(units=100, return_sequences=True),
        Dropout(0.3),
        LSTM(units=50),
        Dropout(0.3),
        Dense(units=25, activation='relu'),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def get_live_price(ticker):
    """Get live stock price"""
    try:
        stock = yf.Ticker(ticker)
        live_price = stock.history(period='1d')['Close'].iloc[-1]
        return float(live_price)
    except Exception as e:
        print(f"Error getting live price: {e}")
        return None

def predict_stock_movement(ticker):
    """Main prediction function with error handling, live price, and advanced analysis"""
    try:
        # Get live price first
        live_price = get_live_price(ticker)
        
        # Perform advanced stock analysis
        advanced_analysis = advanced_stock_analysis(ticker)
        
        stock_prices = cached_stock_data(ticker, get_ttl_hash())
        
        if stock_prices.empty:
            raise ValueError("No data available for this stock")
        if len(stock_prices) < 60:
            raise ValueError(f"Only {len(stock_prices)} data points available (need 60)")
        
        # Prepare data and train model
        X_train, y_train, _, _, scaler = prepare_data(stock_prices)
        model = create_lstm_model((X_train.shape[1], 1))
        model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
        
        # Prepare last 60 data points for prediction
        last_60 = stock_prices[-60:].values.astype('float32')
        last_60_scaled = scaler.transform(last_60.reshape(-1, 1))
        X_pred = np.array([last_60_scaled.flatten()])
        X_pred = np.reshape(X_pred, (X_pred.shape[0], X_pred.shape[1], 1))
        
        # Generate predictions
        predictions = []
        for _ in range(100):
            pred = model.predict(X_pred, verbose=0)
            predictions.append(scaler.inverse_transform(pred)[0][0])
        
        predicted_value = np.mean(predictions)
        last_price = stock_prices.iloc[-1]
        
        movement = "Up" if predicted_value > last_price else "Down"
        confidence = abs((predicted_value - last_price) / last_price) * 100
        
        # Ensure predicted_value is a float
        predicted_value = float(predicted_value)
        
        # Prepare chart data
        dates = stock_prices.index[-60:].strftime('%Y-%m-%d %H:%M').tolist()
        prices = stock_prices[-60:].values.tolist()
        chart_data = (dates, prices)
        
        # Get stock info
        stock_info = get_stock_info(ticker)
        
        return (movement, confidence, predicted_value, last_price,
                chart_data, stock_info, live_price, advanced_analysis)
    
    except Exception as e:
        print(f"Prediction error: {e}")
        stock_info = get_stock_info(ticker)
        return (f"Error: {str(e)}", 0.0, stock_info['current_price'], 
                stock_info['current_price'], ([], []), stock_info, None, None)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        ticker = request.form['ticker']
        stock_name = STOCK_OPTIONS.get(ticker, ticker)
        
        result = predict_stock_movement(ticker)
        (movement, confidence, predicted_value, last_price,
         chart_data, stock_info, live_price, advanced_analysis) = result
        
        return render_template('result.html',
                            ticker=stock_name,
                            movement=movement,
                            confidence=f"{confidence:.2f}" if isinstance(confidence, float) else "N/A",
                            predicted_value=f"{predicted_value:.2f}" if isinstance(predicted_value, float) else "N/A",
                            last_price=f"{last_price:.2f}" if isinstance(last_price, float) else "N/A",
                            live_price=f"{live_price:.2f}" if live_price else "N/A",
                            chart_data=json.dumps(chart_data),
                            sector=stock_info['sector'],
                            current_price=f"{stock_info['current_price']:.2f}" if isinstance(stock_info['current_price'], float) else "N/A",
                            website=stock_info['website'],
                            logo_url=stock_info['logo_url'],
                            timestamp=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            advanced_analysis=advanced_analysis)
    
    return render_template('index.html', stock_options=STOCK_OPTIONS)

if __name__ == '__main__':
    app.run(debug=True)