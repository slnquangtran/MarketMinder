"""Main Streamlit application for stock tracking."""
import streamlit as st
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.portfolio import Portfolio
from src.data.stock_fetcher import StockFetcher
from src.models.lstm_predictor import LSTMPredictor
from src.models.sentiment_analyzer import SentimentAnalyzer
from src.models.anomaly_detector import AnomalyDetector
from src.models.recommender import Recommender
from src.analysis.technical_indicators import TechnicalIndicators
from src.analysis.portfolio_optimizer import PortfolioOptimizer
from src.analysis.backtester import Backtester
from src.visualization.charts import Charts
from src.alerts.email_alerts import EmailAlerts
from src.utils.data_export import DataExporter
from src.utils.config import config

# Page configuration
st.set_page_config(
    page_title="Stock Tracker Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Design System Injection
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Create assets dir if missing and load CSS
assets_path = Path(__file__).parent / "assets" / "style.css"
if assets_path.exists():
    local_css(str(assets_path))
else:
    # Fallback to embedded base if file not found
    st.markdown("""
    <style>
        :root { --primary: #0066FF; --bg-deep: #0A0A0F; }
        body { background-color: var(--bg-deep); color: #E1E1E6; }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = Portfolio()
if 'fetcher' not in st.session_state:
    st.session_state.fetcher = StockFetcher()
if 'predictor' not in st.session_state:
    st.session_state.predictor = LSTMPredictor()

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/200x80/667eea/ffffff?text=Stock+Tracker+Pro", 
             width='stretch')
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "💼 Portfolio", "📊 Analysis", "🤖 AI Predictions", 
         "📰 Sentiment", "🔔 Alerts", "📈 Backtesting", "⚙️ Settings"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Main content
if page == "🏠 Dashboard":
    # Professional Market Hub
    st.markdown('<h1 class="main-header">📈 Quantum Finance Workspace</h1>', unsafe_allow_html=True)
    
    # 1. Ticker Tape Simulation
    ticker_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "BTC-USD", "ETH-USD"]
    tape_content = " | ".join([f"{s}: ${st.session_state.fetcher.get_current_price(s):.2f}" if st.session_state.fetcher.get_current_price(s) else f"{s}: --" for s in ticker_symbols])
    st.markdown(f'<div class="ticker-tape"><marquee scrollamount="5">{tape_content}</marquee></div>', unsafe_allow_html=True)

    # 2. Institutional Performance Cards
    portfolio_summary = st.session_state.portfolio.get_summary()
    daily_gain = portfolio_summary['daily_change_pct'] >= 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="card {"metric-gain" if daily_gain else "metric-loss"}">', unsafe_allow_html=True)
        st.metric("AUM (Assets Under Management)", f"${portfolio_summary['total_value']:,.2f}",
                 f"{portfolio_summary['daily_change_pct']:+.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.metric("Absolute P/L", f"${portfolio_summary['total_profit_loss']:,.2f}",
                 f"{portfolio_summary['total_roi']:+.2f}% ROI")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.metric("Risk Capital Positions", f"{portfolio_summary['num_holdings']}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.metric("Intraday Net", f"${portfolio_summary['daily_change']:+,.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 3. Portfolio Watchlist (Upgraded Table)
    if portfolio_summary['holdings']:
        st.subheader("📊 Active Watchlist & Holdings")
        
        holdings_df = pd.DataFrame(portfolio_summary['holdings'])
        holdings_df = holdings_df[['ticker', 'quantity', 'purchase_price', 'current_price', 
                                   'profit_loss', 'profit_loss_pct']]
        holdings_df.columns = ['Ticker', 'Exposure', 'Entry', 'Spot', 'Net P/L', 'Return %']
        
        # Style dataframe for the mono look
        st.dataframe(holdings_df, use_container_width=True, hide_index=True)
    else:
        st.info("Portfolio inactive. Initialize positions to begin tracking.")
    
    # Quick stock lookup
    st.markdown("---")
    # Quick stock lookup in a professional card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🔍 Institutional Stock Lookup")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        ticker = st.text_input("Enter ticker symbol", placeholder="AAPL", key="dash_lookup").upper()
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        lookup_btn = st.button("Query Market", width='stretch')
    
    if lookup_btn and ticker:
        with st.spinner(f"Querying {ticker}..."):
            price = st.session_state.fetcher.get_current_price(ticker)
            info = st.session_state.fetcher.get_stock_info(ticker)
            
            if price and info:
                c1, c2, c3 = st.columns(3)
                c1.metric("Quote", f"${price:.2f}")
                c2.metric("Institution", info['name'])
                c3.metric("Sector", info['sector'])
            else:
                st.error(f"Data unavailable for {ticker}")
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "💼 Portfolio":
    st.markdown('<h1 class="main-header">💼 Portfolio Management</h1>', 
                unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["⚡ Trade Execution", "📋 Order Book", "📊 Data Export"])
    
    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Open New Position")
        
        col1, col2 = st.columns(2)
        with col1:
            ticker = st.text_input("Ticker Symbol", placeholder="AAPL", key="port_add").upper()
            quantity = st.number_input("Quantity (Shares)", min_value=0.01, value=1.0, step=0.01)
        
        with col2:
            purchase_price = st.number_input("Execution Price ($)", min_value=0.01, value=100.0)
            purchase_date = st.date_input("Settlement Date", value=datetime.now())
        
        if st.button("Execute Order", width='stretch'):
            if ticker:
                success = st.session_state.portfolio.add_stock(
                    ticker, quantity, purchase_price, purchase_date.strftime('%Y-%m-%d')
                )
                if success:
                    st.success(f"ORDER EXECUTED: Buy {quantity} {ticker} @ ${purchase_price:.2f}")
                    st.rerun()
                else:
                    st.error(f"EXECUTION FAILED: Unknown ticker {ticker}")
            else:
                st.warning("Symbol required for execution.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Current Holdings")
        
        summary = st.session_state.portfolio.get_summary()
        
        if summary['holdings']:
            for holding in summary['holdings']:
                with st.expander(f"{holding['ticker']} - {holding['quantity']} shares"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Purchase Price", f"${holding['purchase_price']:.2f}")
                        st.metric("Current Price", f"${holding['current_price']:.2f}")
                    
                    with col2:
                        st.metric("Cost Basis", f"${holding['cost_basis']:.2f}")
                        st.metric("Current Value", f"${holding['current_value']:.2f}")
                    
                    with col3:
                        st.metric("Profit/Loss", f"${holding['profit_loss']:.2f}",
                                 f"{holding['profit_loss_pct']:.2f}%")
                    
                    if st.button(f"Remove {holding['ticker']}", key=f"remove_{holding['ticker']}"):
                        st.session_state.portfolio.remove_stock(holding['ticker'])
                        st.success(f"Removed {holding['ticker']} from portfolio")
                        st.rerun()
        else:
            st.info("No holdings in portfolio.")
    
    with tab3:
        st.subheader("Export Portfolio Data")
        
        exporter = DataExporter()
        
        col1, col2 = st.columns(2)
        with col1:
            export_format = st.selectbox("Export Format", ["CSV", "Excel"])
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Export Portfolio", width='stretch'):
                summary = st.session_state.portfolio.get_summary()
                filepath = exporter.export_portfolio(summary, export_format.lower())
                st.success(f"Portfolio exported to: {filepath}")

elif page == "📊 Analysis":
    st.markdown('<h1 class="main-header">📊 Technical Analysis</h1>', 
                unsafe_allow_html=True)
    
    ticker = st.text_input("Enter ticker symbol for analysis", placeholder="AAPL").upper()
    
    if ticker:
        with st.spinner(f"Analyzing {ticker}..."):
            data = st.session_state.fetcher.get_historical_data(ticker, period='1y')
            
            if data is not None:
                tab1, tab2, tab3 = st.tabs(["Price Chart", "Technical Indicators", "Portfolio Optimization"])
                
                with tab1:
                    st.subheader(f"{ticker} Price Chart")
                    fig = Charts.candlestick_chart(data, ticker)
                    st.plotly_chart(fig, width='stretch')
                
                with tab2:
                    st.subheader("Technical Indicators")
                    fig = Charts.technical_indicators_chart(data, ticker)
                    st.plotly_chart(fig, width='stretch')
                    
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("Technical Signal Dashboard")
                    # Current indicator values in a clean grid
                    data_with_indicators = TechnicalIndicators.add_all_indicators(data)
                    latest = data_with_indicators.iloc[-1]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("RSI (14)", f"{latest['RSI']:.1f}")
                    col2.metric("MACD", f"{latest['MACD']:.2f}")
                    col3.metric("SMA 50", f"${latest['SMA_50']:.1f}")
                    col4.metric("SMA 200", f"${latest['SMA_200']:.1f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with tab3:
                    st.subheader("Portfolio Optimization")
                    
                    tickers_input = st.text_input(
                        "Enter tickers (comma-separated)", 
                        value=ticker,
                        placeholder="AAPL,GOOGL,MSFT"
                    )
                    
                    if st.button("Optimize Portfolio"):
                        tickers = [t.strip().upper() for t in tickers_input.split(',')]
                        
                        with st.spinner("Optimizing..."):
                            optimizer = PortfolioOptimizer()
                            result = optimizer.optimize_sharpe(tickers)
                            
                            if result.get('success'):
                                st.success("Optimization Complete!")
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Expected Return", f"{result['expected_return']*100:.2f}%")
                                with col2:
                                    st.metric("Volatility", f"{result['volatility']*100:.2f}%")
                                with col3:
                                    st.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.2f}")
                                
                                st.subheader("Optimal Weights")
                                weights_df = pd.DataFrame(list(result['weights'].items()), 
                                                         columns=['Ticker', 'Weight'])
                                weights_df['Weight'] = weights_df['Weight'].apply(lambda x: f"{x*100:.2f}%")
                                st.dataframe(weights_df, hide_index=True)
                            else:
                                st.error(result.get('error', 'Optimization failed'))
            else:
                st.error(f"Could not fetch data for {ticker}")

elif page == "🤖 AI Predictions":
    st.markdown('<h1 class="main-header">🤖 AI Price Predictions</h1>', 
                unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Model Configuration")
    col1, col2 = st.columns([3, 1])
    with col1:
        ticker = st.text_input("Predictive Ticker Target", placeholder="AAPL", key="pred_lookup").upper()
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Optimize Weights", width='stretch'):
            with st.spinner(f"Fine-tuning model for {ticker}..."):
                result = st.session_state.predictor.train(ticker)
                if result.get('success'):
                    st.success("Optimization Complete")
                else:
                    st.error("Training Interrupted")
    st.markdown('</div>', unsafe_allow_html=True)
        
    if st.button("Run Analytics Engine", width='stretch'):
        with st.spinner("Generating deep-learning forecast..."):
            predictions = st.session_state.predictor.predict(ticker, days=7)
            
            if predictions:
                st.success("AI Analytics Ready.")
                
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader(f"Forecast Horizon: {ticker}")
                # Show predictions in a clean card
                pred_df = pd.DataFrame(predictions)
                pred_df.columns = ['Date', 'Price Target']
                st.dataframe(pred_df, width='stretch', hide_index=True)
                
                # Chart
                historical_data = st.session_state.fetcher.get_historical_data(ticker, period='3mo')
                if historical_data is not None:
                    fig = Charts.prediction_chart(historical_data, predictions, ticker)
                    st.plotly_chart(fig, width='stretch')
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("Engine requires recent training pulse.")

elif page == "📰 Sentiment":
    st.markdown('<h1 class="main-header">📰 Sentiment Analysis</h1>', 
                unsafe_allow_html=True)
    
    analyzer = SentimentAnalyzer()
    
    if not analyzer.is_configured():
        st.warning("⚠️ Sentiment analysis requires NewsAPI key. Please configure in .env file.")
        st.info("Get your free API key at: https://newsapi.org/")
    else:
        ticker = st.text_input("Enter ticker symbol", placeholder="AAPL").upper()
        company_name = st.text_input("Company name (optional)", placeholder="Apple Inc.")
        
        if st.button("Analyze Sentiment", width='stretch'):
            with st.spinner(f"Analyzing sentiment for {ticker}..."):
                sentiment = analyzer.analyze_stock(ticker, company_name or None)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Sentiment Score", f"{sentiment['sentiment_score']:.2f}")
                with col2:
                    st.metric("Overall Sentiment", sentiment['sentiment_label'].upper())
                with col3:
                    st.metric("Articles Analyzed", sentiment['num_articles'])
                
                # Sentiment breakdown
                st.subheader("Sentiment Breakdown")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Positive", sentiment['positive_count'], delta_color="normal")
                with col2:
                    st.metric("Neutral", sentiment['neutral_count'], delta_color="off")
                with col3:
                    st.metric("Negative", sentiment['negative_count'], delta_color="inverse")
                
                # Recent articles
                if sentiment['articles']:
                    st.subheader("Recent Articles")
                    for article in sentiment['articles'][:5]:
                        with st.expander(f"{article['title']} ({article['source']})"):
                            st.write(f"**Sentiment:** {article['sentiment_label']} ({article['sentiment_score']:.2f})")
                            st.write(f"**Published:** {article['published_at']}")
                            st.write(f"[Read more]({article['url']})")

elif page == "🔔 Alerts":
    st.markdown('<h1 class="main-header">🔔 Alert Management</h1>', 
                unsafe_allow_html=True)
    
    email_alerts = EmailAlerts()
    
    tab1, tab2 = st.tabs(["Configure Alerts", "Send Test Alert"])
    
    with tab1:
        st.subheader("Alert Configuration")
        
        if email_alerts.is_configured():
            st.success("✅ Email alerts are configured")
        else:
            st.warning("⚠️ Email alerts not configured. Add SMTP credentials to .env file")
        
        st.info("Configure alert thresholds in config.yaml")
    
    with tab2:
        st.subheader("Send Test Alert")
        
        if email_alerts.is_configured():
            if st.button("Send Test Email"):
                success = email_alerts.send_email(
                    "Test Alert from Stock Tracker",
                    "<h2>Test Alert</h2><p>Your email alerts are working correctly!</p>"
                )
                if success:
                    st.success("Test email sent successfully!")
                else:
                    st.error("Failed to send test email")
        else:
            st.warning("Email not configured")

elif page == "📈 Backtesting":
    st.markdown('<h1 class="main-header">📈 Strategy Backtesting</h1>', 
                unsafe_allow_html=True)
    
    backtester = Backtester()
    
    col1, col2 = st.columns(2)
    with col1:
        ticker = st.text_input("Ticker Symbol", placeholder="AAPL").upper()
        strategy = st.selectbox("Strategy", ["sma_crossover", "rsi", "macd", "buy_hold"])
    
    with col2:
        start_date = st.date_input("Start Date", value=pd.Timestamp.now() - pd.Timedelta(days=365))
        initial_capital = st.number_input("Initial Capital ($)", value=10000, min_value=100)
    
    if st.button("Run Backtest", width='stretch'):
        with st.spinner(f"Backtesting {strategy} strategy on {ticker}..."):
            result = backtester.run_strategy(
                ticker, strategy, 
                start_date=start_date.strftime('%Y-%m-%d')
            )
            
            if 'error' not in result:
                st.success("Backtest Complete!")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Return", f"{result['total_return_pct']:.2f}%")
                with col2:
                    st.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.2f}")
                with col3:
                    st.metric("Max Drawdown", f"{result['max_drawdown']*100:.2f}%")
                with col4:
                    st.metric("Number of Trades", result['num_trades'])
                
                # Equity curve
                st.subheader("Equity Curve")
                fig = Charts.portfolio_performance_chart(result['equity_curve'])
                st.plotly_chart(fig, use_container_width=True)
                
                # Trades
                if result['trades']:
                    st.subheader("Trade History")
                    trades_df = pd.DataFrame(result['trades'])
                    st.dataframe(trades_df, hide_index=True)
            else:
                st.error(result['error'])

elif page == "⚙️ Settings":
    st.markdown('<h1 class="main-header">⚙️ Settings</h1>', 
                unsafe_allow_html=True)
    
    st.subheader("Configuration Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**API Keys**")
        st.write(f"NewsAPI: {'✅ Configured' if config.has_news_api_key() else '❌ Not configured'}")
        st.write(f"Email: {'✅ Configured' if config.has_email_config() else '❌ Not configured'}")
        st.write(f"SMS: {'✅ Configured' if config.has_sms_config() else '❌ Not configured'}")
    
    with col2:
        st.markdown("**System Info**")
        st.write(f"Portfolio: {st.session_state.portfolio.name}")
        st.write(f"Holdings: {len(st.session_state.portfolio.holdings)}")
    
    st.markdown("---")
    st.subheader("Configuration Files")
    st.info("Edit config.yaml to customize application settings")
    st.info("Edit .env to add API keys (never commit this file!)")
    
    st.markdown("---")
    st.subheader("About")
    st.write("**Stock Tracker Pro** - Comprehensive stock tracking with AI/ML capabilities")
    st.write("Version 1.0.0")
    st.caption("⚠️ Disclaimer: This application is for educational purposes only. Not financial advice.")

# Footer
st.markdown("---")
st.caption("Made with ❤️ using Streamlit | All data provided by yfinance")
