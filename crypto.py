import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Crypto Portfolio Optimizer", layout="wide", page_icon="💰")

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = None
if 'ml_results' not in st.session_state:
    st.session_state.ml_results = None
if 'selected_coins' not in st.session_state:
    st.session_state.selected_coins = []

# CoinGecko API Functions
def fetch_top_coins(limit=100):
    """Fetch top cryptocurrencies from CoinGecko"""
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        'vs_currency': 'usd',
        'order': 'market_cap_desc',
        'per_page': limit,
        'page': 1,
        'sparkline': False
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching coins: {e}")
        return []

def fetch_coin_details(coin_id, max_retries=3):
    """Fetch detailed information for a specific coin with retry logic"""
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=15)
            
            if response.status_code == 429:
                wait_time = 60 * (attempt + 1)
                if attempt < max_retries - 1:
                    st.warning(f"Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    return None
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
                continue
            else:
                st.warning(f"Error fetching details for {coin_id}: {str(e)}")
                return None
    
    return None

def fetch_historical_data(coin_id, days=90, max_retries=3):
    """Fetch historical price data with retry logic"""
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {'vs_currency': 'usd', 'days': days}
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 429:
                wait_time = 60 * (attempt + 1)
                if attempt < max_retries - 1:
                    st.warning(f"Rate limited for {coin_id}. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    st.warning(f"Skipping {coin_id} after {max_retries} attempts")
                    return None
            
            response.raise_for_status()
            data = response.json()
            prices = data['prices']
            df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
                continue
            else:
                st.warning(f"Error fetching historical data for {coin_id}: {str(e)}")
                return None
    
    return None

def check_supply_ratio(coin_data):
    """Check if circulating supply is less than 50% of max supply"""
    circ_supply = coin_data.get('circulating_supply', 0)
    max_supply = coin_data.get('max_supply', 0)
    
    if max_supply is None or max_supply == 0:
        return True, 100.0
    
    ratio = (circ_supply / max_supply) * 100
    return ratio >= 50, ratio

def process_coin_data(coins_data):
    """Process and filter coins based on supply criteria"""
    processed_coins = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Add rate limit counter
    api_calls = 0
    batch_size = 10
    
    for idx, coin in enumerate(coins_data):
        status_text.text(f"Processing {coin['name']} ({idx+1}/{len(coins_data)}) - API calls: {api_calls}")
        progress_bar.progress((idx + 1) / len(coins_data))
        
        if coin['symbol'].upper() in ['BTC', 'USDT']:
            continue
        
        # Rate limiting: pause every batch_size calls
        if api_calls > 0 and api_calls % batch_size == 0:
            wait_time = 65  # Wait 65 seconds to reset rate limit
            status_text.text(f"⏳ Rate limit protection: Waiting {wait_time}s... ({api_calls} API calls made)")
            time.sleep(wait_time)
        
        coin_details = fetch_coin_details(coin['id'])
        api_calls += 1
        
        if coin_details:
            is_valid, supply_ratio = check_supply_ratio(coin_details['market_data'])
            
            processed_coins.append({
                'id': coin['id'],
                'symbol': coin['symbol'].upper(),
                'name': coin['name'],
                'current_price': coin['current_price'],
                'market_cap': coin['market_cap'],
                'total_volume': coin['total_volume'],
                'price_change_24h': coin.get('price_change_percentage_24h', 0),
                'price_change_7d': coin.get('price_change_percentage_7d_in_currency', 0),
                'circulating_supply': coin_details['market_data'].get('circulating_supply', 0),
                'max_supply': coin_details['market_data'].get('max_supply', 0),
                'supply_ratio': supply_ratio,
                'is_valid': is_valid,
                'ath': coin_details['market_data'].get('ath', {}).get('usd', 0),
                'atl': coin_details['market_data'].get('atl', {}).get('usd', 0),
            })
        
        time.sleep(1.5)  # Increased delay between calls
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(processed_coins)

def perform_eda(df):
    """Perform Exploratory Data Analysis"""
    st.markdown("<h2 class='sub-header'>📊 Exploratory Data Analysis</h2>", unsafe_allow_html=True)
    
    # Filter out coins with infinite max supply (max_supply = 0 or None)
    valid_df = df[df['is_valid'] == True].copy()
    valid_df = valid_df[(valid_df['max_supply'] > 0) & (valid_df['max_supply'].notna())].copy()
    
    st.info(f"ℹ️ Filtered out coins with infinite/unlimited max supply for better analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Coins Analyzed", len(df))
    with col2:
        st.metric("Valid Coins (>50% Supply)", len(valid_df))
    with col3:
        st.metric("Rejected Coins", len(df) - len(valid_df))
    with col4:
        st.metric("Average Supply Ratio", f"{valid_df['supply_ratio'].mean():.1f}%")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(valid_df, x='current_price', nbins=30, 
                          title='Price Distribution (Valid Coins)',
                          labels={'current_price': 'Price (USD)'})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(valid_df, x='market_cap', y='total_volume',
                        size='current_price', hover_data=['name', 'symbol'],
                        title='Market Cap vs Trading Volume',
                        labels={'market_cap': 'Market Cap', 'total_volume': 'Volume'},
                        log_x=True, log_y=True)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    fig = px.bar(valid_df.nlargest(20, 'market_cap'), 
                 x='symbol', y='supply_ratio',
                 title='Supply Ratio for Top 20 Valid Coins',
                 labels={'supply_ratio': 'Supply Ratio (%)', 'symbol': 'Coin'},
                 color='supply_ratio',
                 color_continuous_scale='Viridis')
    fig.add_hline(y=50, line_dash="dash", line_color="red", 
                  annotation_text="50% Threshold")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    numeric_cols = ['current_price', 'market_cap', 'total_volume', 
                   'price_change_24h', 'price_change_7d', 'supply_ratio']
    corr_matrix = valid_df[numeric_cols].corr()
    
    fig = px.imshow(corr_matrix, 
                    title='Feature Correlation Heatmap',
                    labels=dict(color="Correlation"),
                    x=numeric_cols, y=numeric_cols,
                    color_continuous_scale='RdBu_r',
                    aspect="auto")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📈 Summary Statistics")
    st.dataframe(valid_df[numeric_cols].describe(), use_container_width=True)
    
    return valid_df

def perform_clustering(df):
    """Perform K-Means clustering on coins"""
    st.markdown("<h2 class='sub-header'>🎯 Coin Clustering Analysis</h2>", unsafe_allow_html=True)
    
    features = ['current_price', 'market_cap', 'total_volume', 
               'price_change_24h', 'price_change_7d', 'supply_ratio']
    
    X = df[features].fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    inertias = []
    K_range = range(2, min(11, len(df)//3))
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(K_range), y=inertias, mode='lines+markers'))
    fig.update_layout(title='Elbow Method - Optimal Clusters',
                     xaxis_title='Number of Clusters',
                     yaxis_title='Inertia',
                     height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(df, x='market_cap', y='total_volume',
                        color='cluster', hover_data=['name', 'symbol'],
                        title='Clusters: Market Cap vs Volume',
                        log_x=True, log_y=True)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(df, x='price_change_24h', y='price_change_7d',
                        color='cluster', hover_data=['name', 'symbol'],
                        title='Clusters: Price Changes',
                        size='market_cap')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📊 Cluster Characteristics")
    cluster_summary = df.groupby('cluster')[features].mean()
    st.dataframe(cluster_summary, use_container_width=True)
    
    cluster_counts = df['cluster'].value_counts().sort_index()
    fig = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                 title='Number of Coins per Cluster',
                 labels={'x': 'Cluster', 'y': 'Count'})
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    return df

def train_ml_models(coin_id, coin_name):
    """Train ML models to predict future prices"""
    hist_data = fetch_historical_data(coin_id, days=90)
    
    if hist_data is None or len(hist_data) < 30:
        return None
    
    hist_data['day'] = (hist_data['date'] - hist_data['date'].min()).dt.days
    hist_data['price_ma7'] = hist_data['price'].rolling(window=7, min_periods=1).mean()
    hist_data['price_ma14'] = hist_data['price'].rolling(window=14, min_periods=1).mean()
    hist_data['price_std7'] = hist_data['price'].rolling(window=7, min_periods=1).std()
    hist_data['price_change'] = hist_data['price'].pct_change()
    hist_data['volatility'] = hist_data['price_change'].rolling(window=7, min_periods=1).std()
    
    hist_data = hist_data.fillna(method='bfill').fillna(0)
    
    features = ['day', 'price_ma7', 'price_ma14', 'price_std7', 'volatility']
    X = hist_data[features]
    y = hist_data['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        results[name] = {
            'model': model,
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'predictions': y_pred,
            'actual': y_test
        }
    
    future_days = np.arange(hist_data['day'].max() + 1, hist_data['day'].max() + 31)
    last_ma7 = hist_data['price_ma7'].iloc[-1]
    last_ma14 = hist_data['price_ma14'].iloc[-1]
    last_std7 = hist_data['price_std7'].iloc[-1]
    last_volatility = hist_data['volatility'].iloc[-1]
    
    future_features = pd.DataFrame({
        'day': future_days,
        'price_ma7': [last_ma7] * 30,
        'price_ma14': [last_ma14] * 30,
        'price_std7': [last_std7] * 30,
        'volatility': [last_volatility] * 30
    })
    
    best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
    best_model = results[best_model_name]['model']
    future_predictions = best_model.predict(future_features)
    
    results['future_predictions'] = future_predictions
    results['future_days'] = future_days
    results['best_model'] = best_model_name
    results['historical_data'] = hist_data
    
    return results

def display_ml_results(coin_data, ml_results):
    """Display ML model results"""
    st.markdown(f"#### 🤖 ML Predictions for {coin_data['name']} ({coin_data['symbol']})")
    
    col1, col2, col3 = st.columns(3)
    
    best_model = ml_results['best_model']
    with col1:
        st.metric("Best Model", best_model)
    with col2:
        st.metric("R² Score", f"{ml_results[best_model]['r2']:.4f}")
    with col3:
        st.metric("RMSE", f"${ml_results[best_model]['rmse']:.4f}")
    
    st.markdown("##### Model Performance Comparison")
    model_comparison = pd.DataFrame({
        'Model': ['Linear Regression', 'Random Forest', 'Gradient Boosting'],
        'R² Score': [ml_results[m]['r2'] for m in ['Linear Regression', 'Random Forest', 'Gradient Boosting']],
        'RMSE': [ml_results[m]['rmse'] for m in ['Linear Regression', 'Random Forest', 'Gradient Boosting']],
        'MAE': [ml_results[m]['mae'] for m in ['Linear Regression', 'Random Forest', 'Gradient Boosting']]
    })
    st.dataframe(model_comparison, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        hist_data = ml_results['historical_data']
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_data['date'], y=hist_data['price'],
                                mode='lines', name='Actual Price'))
        fig.add_trace(go.Scatter(x=hist_data['date'], y=hist_data['price_ma7'],
                                mode='lines', name='7-day MA', line=dict(dash='dash')))
        
        fig.update_layout(title='Historical Price with Moving Average',
                         xaxis_title='Date', yaxis_title='Price (USD)',
                         height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        current_price = coin_data['current_price']
        future_dates = pd.date_range(start=hist_data['date'].max() + timedelta(days=1), 
                                     periods=30)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=future_dates, y=ml_results['future_predictions'],
                                mode='lines+markers', name='Predicted Price'))
        fig.add_hline(y=current_price, line_dash="dash", line_color="red",
                     annotation_text=f"Current: ${current_price:.2f}")
        
        fig.update_layout(title='30-Day Price Prediction',
                         xaxis_title='Date', yaxis_title='Predicted Price (USD)',
                         height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    predicted_change = ((ml_results['future_predictions'][-1] - current_price) / current_price) * 100
    
    # Calculate 1-year and 5-year projections
    predicted_1year = current_price * (1 + (predicted_change / 100)) ** 12
    predicted_5year = current_price * (1 + (predicted_change / 100)) ** 60
    
    return_1year = ((predicted_1year - current_price) / current_price) * 100
    return_5year = ((predicted_5year - current_price) / current_price) * 100
    
    st.markdown("### 📈 Projected Returns")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("30-Day Return", f"{predicted_change:.2f}%")
    with col2:
        st.metric("1-Year Projected Return", f"{return_1year:.2f}%", 
                 delta=f"${predicted_1year:.4f}")
    with col3:
        st.metric("5-Year Projected Return", f"{return_5year:.2f}%",
                 delta=f"${predicted_5year:.4f}")
    
    return predicted_change

def optimize_portfolio(valid_coins, ml_predictions, altcoin_budget):
    """Optimize portfolio allocation based on ML predictions"""
    st.markdown("<h2 class='sub-header'>💼 Portfolio Optimization</h2>", unsafe_allow_html=True)
    
    scored_coins = []
    
    for coin_id, prediction_data in ml_predictions.items():
        coin_info = valid_coins[valid_coins['id'] == coin_id].iloc[0]
        
        predicted_return = prediction_data['predicted_change']
        r2_score_val = prediction_data['r2_score']
        market_cap_score = np.log(coin_info['market_cap']) / 30
        volume_score = np.log(coin_info['total_volume']) / 25
        
        total_score = (
            predicted_return * 0.4 +
            r2_score_val * 20 +
            market_cap_score * 0.2 +
            volume_score * 0.2
        )
        
        scored_coins.append({
            'id': coin_id,
            'name': coin_info['name'],
            'symbol': coin_info['symbol'],
            'predicted_return': predicted_return,
            'r2_score': r2_score_val,
            'total_score': total_score,
            'current_price': coin_info['current_price']
        })
    
    scored_df = pd.DataFrame(scored_coins).sort_values('total_score', ascending=False)
    top_coins = scored_df.head(10)
    
    total_score = top_coins['total_score'].sum()
    top_coins['allocation'] = (top_coins['total_score'] / total_score) * altcoin_budget
    top_coins['allocation_pct'] = (top_coins['allocation'] / altcoin_budget) * 100
    
    st.markdown("### 📊 Recommended Altcoin Allocation")
    
    display_df = top_coins[['symbol', 'name', 'predicted_return', 'r2_score', 
                           'allocation', 'allocation_pct']].copy()
    display_df.columns = ['Symbol', 'Name', 'Predicted Return (%)', 'R² Score', 
                         'Allocation ($)', 'Allocation (%)']
    st.dataframe(display_df, use_container_width=True)
    
    fig = px.pie(top_coins, values='allocation', names='symbol',
                 title='Portfolio Allocation Distribution',
                 hole=0.4)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    expected_return = (top_coins['predicted_return'] * top_coins['allocation_pct'] / 100).sum()
    
    # Calculate 1-year and 5-year portfolio projections
    expected_1year = altcoin_budget * (1 + (expected_return / 100)) ** 12
    expected_5year = altcoin_budget * (1 + (expected_return / 100)) ** 60
    
    return_1year = ((expected_1year - altcoin_budget) / altcoin_budget) * 100
    return_5year = ((expected_5year - altcoin_budget) / altcoin_budget) * 100
    
    st.markdown("### 📈 Portfolio Expected Returns")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("30-Day Portfolio Return", f"{expected_return:.2f}%")
    with col2:
        st.metric("1-Year Projected Return", f"{return_1year:.2f}%",
                 delta=f"${expected_1year:,.2f}")
    with col3:
        st.metric("5-Year Projected Return", f"{return_5year:.2f}%",
                 delta=f"${expected_5year:,.2f}")
    
    return top_coins

def backtest_strategy(top_coins, investment_capital):
    """Backtest the strategy over the last 3 months"""
    st.markdown("<h2 class='sub-header'>📈 Backtesting Simulation (Last 3 Months)</h2>", unsafe_allow_html=True)
    
    st.info("Comparing ML-Optimized Strategy vs Simple HODL Strategy")
    
    backtest_results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (_, coin) in enumerate(top_coins.iterrows()):
        status_text.text(f"Backtesting {coin['symbol']} ({idx+1}/{len(top_coins)})")
        
        hist_data = fetch_historical_data(coin['id'], days=90)
        if hist_data is not None and len(hist_data) > 0:
            initial_price = hist_data['price'].iloc[0]
            final_price = hist_data['price'].iloc[-1]
            returns = ((final_price - initial_price) / initial_price) * 100
            
            hist_data['returns'] = hist_data['price'].pct_change()
            volatility = hist_data['returns'].std() * np.sqrt(365)
            
            rolling_max = hist_data['price'].expanding().max()
            drawdown = (hist_data['price'] - rolling_max) / rolling_max
            max_drawdown = drawdown.min() * 100
            
            backtest_results.append({
                'symbol': coin['symbol'],
                'initial_price': initial_price,
                'final_price': final_price,
                'returns': returns,
                'allocation': coin['allocation'],
                'volatility': volatility * 100,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': (returns / 100) / volatility if volatility > 0 else 0
            })
        progress_bar.progress((idx + 1) / len(top_coins))
        time.sleep(0.5)
    
    progress_bar.empty()
    status_text.empty()
    
    if not backtest_results:
        st.warning("Unable to fetch backtest data")
        return
    
    backtest_df = pd.DataFrame(backtest_results)
    
    backtest_df['weighted_returns'] = (backtest_df['returns'] * 
                                       backtest_df['allocation'] / 
                                       backtest_df['allocation'].sum())
    
    ml_strategy_return = backtest_df['weighted_returns'].sum()
    
    equal_allocation = 1 / len(backtest_df)
    hodl_return = (backtest_df['returns'] * equal_allocation).sum()
    
    portfolio_volatility = np.sqrt((backtest_df['volatility']**2 * 
                                    (backtest_df['allocation'] / backtest_df['allocation'].sum())**2).sum())
    portfolio_sharpe = ml_strategy_return / portfolio_volatility if portfolio_volatility > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("ML Strategy Return", f"{ml_strategy_return:.2f}%", 
                 delta=f"{ml_strategy_return:.2f}%")
    with col2:
        st.metric("HODL Strategy Return", f"{hodl_return:.2f}%",
                 delta=f"{hodl_return:.2f}%")
    with col3:
        outperformance = ml_strategy_return - hodl_return
        st.metric("Outperformance", f"{outperformance:.2f}%",
                 delta=f"{outperformance:.2f}%")
    with col4:
        st.metric("Portfolio Sharpe Ratio", f"{portfolio_sharpe:.2f}")
    
    st.markdown("### ⚠️ Risk Metrics by Coin")
    risk_df = backtest_df[['symbol', 'returns', 'volatility', 'max_drawdown', 'sharpe_ratio']].copy()
    risk_df.columns = ['Symbol', 'Returns (%)', 'Volatility (%)', 'Max Drawdown (%)', 'Sharpe Ratio']
    st.dataframe(risk_df, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Individual Coin Returns")
        fig = px.bar(backtest_df, x='symbol', y='returns',
                     title='3-Month Returns by Coin',
                     labels={'returns': 'Returns (%)', 'symbol': 'Coin'},
                     color='returns',
                     color_continuous_scale='RdYlGn')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📉 Risk vs Return")
        fig = px.scatter(backtest_df, x='volatility', y='returns',
                        size='allocation', hover_data=['symbol'],
                        title='Risk-Return Trade-off',
                        labels={'volatility': 'Volatility (%)', 'returns': 'Returns (%)'},
                        color='sharpe_ratio',
                        color_continuous_scale='Viridis')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    strategies = pd.DataFrame({
        'Strategy': ['ML-Optimized', 'HODL (Equal Weight)'],
        'Returns': [ml_strategy_return, hodl_return],
        'Type': ['ML', 'HODL']
    })
    
    fig = px.bar(strategies, x='Strategy', y='Returns',
                 title='Strategy Performance Comparison',
                 color='Type',
                 color_discrete_map={'ML': '#00CC96', 'HODL': '#EF553B'})
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    ml_final_value = investment_capital * (1 + ml_strategy_return / 100)
    hodl_final_value = investment_capital * (1 + hodl_return / 100)
    
    st.markdown("### 💰 Final Portfolio Values")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("ML Strategy Final Value", f"${ml_final_value:,.2f}",
                 delta=f"${ml_final_value - investment_capital:,.2f}")
    with col2:
        st.metric("HODL Strategy Final Value", f"${hodl_final_value:,.2f}",
                 delta=f"${hodl_final_value - investment_capital:,.2f}")
    with col3:
        profit_diff = ml_final_value - hodl_final_value
        st.metric("Additional Profit (ML)", f"${profit_diff:,.2f}",
                 delta=f"{(profit_diff/investment_capital)*100:.2f}%")
    
    st.markdown("### 🎯 Win/Loss Analysis")
    winning_coins = len(backtest_df[backtest_df['returns'] > 0])
    losing_coins = len(backtest_df[backtest_df['returns'] <= 0])
    win_rate = (winning_coins / len(backtest_df)) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Winning Coins", winning_coins)
    with col2:
        st.metric("Losing Coins", losing_coins)
    with col3:
        st.metric("Win Rate", f"{win_rate:.1f}%")

def main():
    st.markdown("<h1 class='main-header'>💰 ML-Based Crypto Portfolio Optimizer</h1>", unsafe_allow_html=True)
    
    st.sidebar.title("🎛️ Configuration")
    investment_capital = st.sidebar.number_input(
        "Investment Capital ($)", 
        min_value=100.0, 
        value=10000.0, 
        step=100.0
    )
    
    num_coins = st.sidebar.slider("Number of Coins to Analyze", 20, 100, 50)
    
    st.sidebar.warning("⚠️ CoinGecko Free API Limits:\n- ~10-50 calls/minute\n- Reduce coin count if errors occur")
    
    page = st.sidebar.radio("Navigation", 
                           ["Data Collection", "EDA & ML Analysis", 
                            "Portfolio Optimization", "Backtesting"])
    
    btc_allocation = investment_capital * 0.20
    usdt_allocation = investment_capital * 0.20
    reserve = investment_capital * 0.10
    altcoin_budget = investment_capital * 0.50
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💵 Fund Allocation")
    st.sidebar.write(f"**BTC (20%):** ${btc_allocation:,.2f}")
    st.sidebar.write(f"**USDT (20%):** ${usdt_allocation:,.2f}")
    st.sidebar.write(f"**Reserve (10%):** ${reserve:,.2f}")
    st.sidebar.write(f"**Altcoins (50%):** ${altcoin_budget:,.2f}")
    
    if page == "Data Collection":
        st.markdown("<h2 class='sub-header'>🔍 Data Collection & Filtering</h2>", unsafe_allow_html=True)
        
        if st.button("Fetch & Process Coins", type="primary"):
            with st.spinner("Fetching data from CoinGecko..."):
                coins_data = fetch_top_coins(num_coins)
                
                if coins_data:
                    st.success(f"Fetched {len(coins_data)} coins!")
                    df = process_coin_data(coins_data)
                    st.session_state.portfolio_data = df
                    
                    st.markdown("### 📋 Processed Coins Data")
                    st.dataframe(df, use_container_width=True)
                    
                    valid_count = len(df[df['is_valid'] == True])
                    rejected_count = len(df[df['is_valid'] == False])
                    
                    st.success(f"✅ Valid Coins: {valid_count} | ❌ Rejected Coins: {rejected_count}")
                else:
                    st.error("Failed to fetch coin data. Please try again.")
        
        if st.session_state.portfolio_data is not None:
            st.info(f"Data loaded: {len(st.session_state.portfolio_data)} coins")
    
    elif page == "EDA & ML Analysis":
        if st.session_state.portfolio_data is not None:
            st.markdown("<h2 class='sub-header'>📊 EDA & Machine Learning Analysis</h2>", unsafe_allow_html=True)
            
            st.info(f"Ready to analyze {len(st.session_state.portfolio_data)} coins collected from Data Collection")
            
            # Show what will happen
            valid_count = len(st.session_state.portfolio_data[st.session_state.portfolio_data['is_valid'] == True])
            st.markdown(f"""
            ### What will happen:
            1. **Exploratory Data Analysis (EDA)** - Statistical analysis and visualizations
            2. **K-Means Clustering** - Group similar coins together
            3. **Filter coins** - Remove coins with infinite max supply
            4. **ML Training** - Train 3 models (Linear Regression, Random Forest, Gradient Boosting) on **ALL {valid_count} valid coins**
            5. **Price Predictions** - 30-day, 1-year, and 5-year projections
            
            ⏱️ **Estimated Time:** ~{valid_count * 2} minutes (due to API rate limits)
            """)
            
            if st.button("Run EDA & ML Analysis", type="primary", use_container_width=True):
                # Step 1: Perform EDA
                df = st.session_state.portfolio_data
                valid_df = perform_eda(df)
                
                # Perform clustering
                st.markdown("---")
                clustered_df = perform_clustering(valid_df)
                st.session_state.portfolio_data = clustered_df
                
                st.success("✅ EDA and Clustering completed!")
                
                # Step 2: Run ML on ALL coins from EDA
                st.markdown("---")
                st.markdown("<h2 class='sub-header'>🤖 Machine Learning Predictions</h2>", unsafe_allow_html=True)
                
                st.info(f"Training ML models on ALL {len(clustered_df)} coins from EDA...")
                st.warning(f"⚠️ This will fetch historical data for {len(clustered_df)} coins. Estimated time: ~{len(clustered_df) * 2} minutes")
                
                ml_results_dict = {}
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                api_calls = 0
                
                for idx, (_, coin) in enumerate(clustered_df.iterrows()):
                    status_text.text(f"Training models for {coin['name']} ({idx+1}/{len(clustered_df)}) - API calls: {api_calls}")
                    
                    # Rate limiting protection
                    if api_calls > 0 and api_calls % 5 == 0:
                        wait_time = 65
                        status_text.text(f"⏳ Rate limit protection: Waiting {wait_time}s...")
                        time.sleep(wait_time)
                    
                    ml_result = train_ml_models(coin['id'], coin['name'])
                    api_calls += 1
                    
                    if ml_result:
                        with st.expander(f"📊 {coin['name']} ({coin['symbol']})"):
                            predicted_change = display_ml_results(coin, ml_result)
                            
                            ml_results_dict[coin['id']] = {
                                'coin_data': coin,
                                'ml_results': ml_result,
                                'predicted_change': predicted_change,
                                'r2_score': ml_result[ml_result['best_model']]['r2']
                            }
                    
                    progress_bar.progress((idx + 1) / len(clustered_df))
                    time.sleep(2)
                
                st.session_state.ml_results = ml_results_dict
                progress_bar.empty()
                status_text.empty()
                
                st.success(f"✅ ML analysis completed for {len(ml_results_dict)} coins!")
                
                # Summary of predictions
                st.markdown("### 📊 ML Predictions Summary")
                summary_data = []
                for coin_id, data in ml_results_dict.items():
                    summary_data.append({
                        'Symbol': data['coin_data']['symbol'],
                        'Name': data['coin_data']['name'],
                        'Current Price': f"${data['coin_data']['current_price']:.4f}",
                        'Predicted Change (%)': f"{data['predicted_change']:.2f}%",
                        'R² Score': f"{data['r2_score']:.4f}",
                        'Best Model': data['ml_results']['best_model']
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
        else:
            st.warning("Please complete the Data Collection step first.")
            st.info("👈 Go to 'Data Collection' in the sidebar and click 'Fetch & Process Coins'")
    
    elif page == "Portfolio Optimization":
        if st.session_state.ml_results is not None:
            valid_df = st.session_state.portfolio_data[st.session_state.portfolio_data['is_valid'] == True]
            
            top_coins = optimize_portfolio(valid_df, st.session_state.ml_results, altcoin_budget)
            st.session_state.selected_coins = top_coins
            
            st.markdown("### 💼 Complete Portfolio Allocation")
            
            final_portfolio = pd.DataFrame({
                'Asset': ['Bitcoin (BTC)', 'Tether (USDT)', 'Reserve'] + top_coins['symbol'].tolist(),
                'Allocation ($)': [btc_allocation, usdt_allocation, reserve] + top_coins['allocation'].tolist(),
                'Allocation (%)': [20, 20, 10] + top_coins['allocation_pct'].tolist()
            })
            
            st.dataframe(final_portfolio, use_container_width=True)
            
            fig = px.pie(final_portfolio, values='Allocation ($)', names='Asset',
                        title='Complete Portfolio Distribution',
                        hole=0.4)
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### ⚠️ Risk Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                avg_r2 = top_coins['r2_score'].mean()
                st.metric("Average Model Accuracy (R²)", f"{avg_r2:.4f}")
            
            with col2:
                diversification = len(top_coins)
                st.metric("Number of Altcoins", diversification)
            
            # Add 1-year and 5-year projections
            st.markdown("### 🚀 Long-term Portfolio Projections")
            st.info("Based on compounded monthly returns from ML predictions")
            
            expected_return = (top_coins['predicted_return'] * top_coins['allocation_pct'] / 100).sum()
            
            portfolio_1year = investment_capital * (1 + (expected_return / 100)) ** 12
            portfolio_5year = investment_capital * (1 + (expected_return / 100)) ** 60
            
            return_1year = ((portfolio_1year - investment_capital) / investment_capital) * 100
            return_5year = ((portfolio_5year - investment_capital) / investment_capital) * 100
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("1-Year Projected Value", f"${portfolio_1year:,.2f}",
                         delta=f"{return_1year:.2f}% return")
            with col2:
                st.metric("5-Year Projected Value", f"${portfolio_5year:,.2f}",
                         delta=f"{return_5year:.2f}% return")
            
        else:
            st.warning("Please complete the EDA & ML Analysis step first.")
    
    elif page == "Backtesting":
        if st.session_state.selected_coins is not None and len(st.session_state.selected_coins) > 0:
            backtest_strategy(st.session_state.selected_coins, altcoin_budget)
        else:
            st.warning("Please complete the Portfolio Optimization step first.")

if __name__ == "__main__":
    main()