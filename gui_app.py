"""Main Desktop GUI application for stock tracking."""
import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.portfolio import Portfolio
from src.data.stock_fetcher import StockFetcher
from src.visualization.gui_components import GUICharts
from src.models.lstm_predictor import LSTMPredictor
from src.models.ensemble_predictor import EnsemblePredictor
from src.visualization.plotly_dashboard import PlotlyDashboard
from src.models.sentiment_analyzer import SentimentAnalyzer
from src.models.recommender import Recommender
from src.utils.config import config
import webbrowser

# Set appearance mode and theme
ctk.set_appearance_mode("Dark")
# Custom Color Palette
COLORS = {
    "bg_deep": "#0A0A0F",
    "surface": "#16161E",
    "primary": "#0066FF",
    "accent": "#3B82F6",
    "success": "#00C853",
    "danger": "#FF3D00",
    "text": "#E1E1E6",
    "text_dark": "#94A3B8"
}

class StockTrackerApp(ctk.CTk):
    """Main application class for the stock tracking desktop app."""
    
    def __init__(self):
        super().__init__()
        
        # Configure window
        self.title("Stock Tracker Pro | Institutional Intelligence")
        self.geometry("1200x850")
        self.configure(fg_color=COLORS["bg_deep"])
        
        # Initialize data models
        self.portfolio = Portfolio()
        self.fetcher = StockFetcher()
        self.predictor = EnsemblePredictor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.recommender = Recommender()
        
        # Create layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Sidebar
        self.sidebar_frame = ctk.CTkFrame(self, width=220, corner_radius=0, fg_color=COLORS["surface"], border_width=1, border_color="#1E293B")
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(9, weight=1)
        
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="STOCK TRACKER", 
                                       font=ctk.CTkFont(size=22, weight="bold"), text_color=COLORS["primary"])
        self.logo_label.grid(row=0, column=0, padx=20, pady=(30, 5))
        self.sub_logo = ctk.CTkLabel(self.sidebar_frame, text="QUANTUM EDITION", 
                                     font=ctk.CTkFont(size=10, weight="bold"), text_color=COLORS["text_dark"])
        self.sub_logo.grid(row=1, column=0, padx=20, pady=(0, 20))
        
        # Sidebar buttons
        self.dashboard_button = self.create_sidebar_button("🏠 Dashboard", 2, self.show_dashboard)
        self.portfolio_button = self.create_sidebar_button("💼 Portfolio", 3, self.show_portfolio)
        self.analysis_button = self.create_sidebar_button("📊 Analysis", 4, self.show_analysis)
        self.predictions_button = self.create_sidebar_button("🤖 Predictions", 5, self.show_predictions)
        self.sentiment_button = self.create_sidebar_button("📰 Sentiment", 6, self.show_sentiment)
        self.alerts_button = self.create_sidebar_button("🔔 Alerts", 7, self.show_alerts)
        self.backtest_button = self.create_sidebar_button("📈 Backtesting", 8, self.show_backtesting)
        self.settings_button = self.create_sidebar_button("⚙️ Settings", 9, self.show_settings)
        
        self.appearance_mode_label = ctk.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=10, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Dark", "Light", "System"],
                                                            command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=11, column=0, padx=20, pady=(10, 20))
        
        # Main content area
        self.main_content_frame = ctk.CTkFrame(self, corner_radius=15, fg_color=COLORS["bg_deep"], border_width=0)
        self.main_content_frame.grid(row=0, column=1, sticky="nsew", padx=15, pady=15)
        self.main_content_frame.grid_columnconfigure(0, weight=1)
        self.main_content_frame.grid_rowconfigure(0, weight=1)
        
        # Initialize pages
        self.pages = {}
        self.create_pages()
        
        # Show default page
        self.show_dashboard()

    def create_sidebar_button(self, text, row, command):
        button = ctk.CTkButton(self.sidebar_frame, text=text, corner_radius=8, height=45, border_spacing=10,
                               fg_color="transparent", text_color="#94A3B8", hover_color="#1E293B",
                               font=ctk.CTkFont(size=14),
                               anchor="w", command=command)
        button.grid(row=row, column=0, sticky="ew", padx=10, pady=2)
        return button

    def create_pages(self):
        """Pre-create placeholder frames for each page."""
        for page in ["Dashboard", "Portfolio", "Analysis", "AI Predictions", "Sentiment", "Alerts", "Backtesting", "Settings"]:
            self.pages[page] = ctk.CTkFrame(self.main_content_frame, fg_color=COLORS["bg_deep"])
            self.pages[page].grid(row=0, column=0, sticky="nsew")
            self.pages[page].grid_columnconfigure(0, weight=1)

    def show_page(self, page_name):
        # Bring the specified page to the front.
        self.pages[page_name].tkraise()
        
        # Update button colors
        for name, button in [("Dashboard", self.dashboard_button), ("Portfolio", self.portfolio_button),
                             ("Analysis", self.analysis_button), ("AI Predictions", self.predictions_button),
                             ("Sentiment", self.sentiment_button), ("Alerts", self.alerts_button),
                             ("Backtesting", self.backtest_button), ("Settings", self.settings_button)]:
            if name == page_name:
                button.configure(fg_color="#1E293B", text_color="white")
            else:
                button.configure(fg_color="transparent", text_color="#94A3B8")

    def show_dashboard(self):
        self.show_page("Dashboard")
        for widget in self.pages["Dashboard"].winfo_children():
            widget.destroy()
            
        header = ctk.CTkLabel(self.pages["Dashboard"], text="INSTITUTIONAL WORKSPACE", 
                               font=ctk.CTkFont(size=24, weight="bold"), text_color=COLORS["text"])
        header.pack(pady=(20, 10))
        
        # Live Ticker Tape Simulator / Subtitle
        subtitle = ctk.CTkLabel(self.pages["Dashboard"], text="Real-time Quantitative Analytics & Portfolio Governance", 
                                font=ctk.CTkFont(size=12), text_color=COLORS["text_dark"])
        subtitle.pack(pady=(0, 20))
        
        stats_frame = ctk.CTkFrame(self.pages["Dashboard"], fg_color="transparent")
        stats_frame.pack(fill="x", padx=30)
        
        summary = self.portfolio.get_summary()
        self.create_stat_card(stats_frame, "AUM TOTAL VALUE", f"${summary['total_value']:,.2f}", 0)
        self.create_stat_card(stats_frame, "P/L REALIZED", f"${summary['total_profit_loss']:,.2f}", 1)
        self.create_stat_card(stats_frame, "ROI PERFORMANCE", f"{summary['total_roi']:.2f}%", 2)
        
        # Status area
        self.status_label = ctk.CTkLabel(self.pages["Dashboard"], text="System Status: Operational", 
                                        font=ctk.CTkFont(size=11), text_color=COLORS["success"])
        self.status_label.pack(pady=15)
        
        # Enhanced Lookup
        lookup_card = ctk.CTkFrame(self.pages["Dashboard"], fg_color=COLORS["surface"], corner_radius=12, border_width=1, border_color="#1E293B")
        lookup_card.pack(fill="x", padx=30, pady=10)
        
        ctk.CTkLabel(lookup_card, text="QUICK MARKET LOOKUP", font=ctk.CTkFont(size=12, weight="bold"), text_color=COLORS["text_dark"]).pack(pady=(10, 5))
        
        entry_container = ctk.CTkFrame(lookup_card, fg_color="transparent")
        entry_container.pack(fill="x", padx=20, pady=(0, 15))
        
        self.lookup_entry = ctk.CTkEntry(entry_container, placeholder_text="Symbol (e.g. BTC-USD)", height=40, font=ctk.CTkFont(size=14))
        self.lookup_entry.pack(side="left", padx=(0, 10), expand=True, fill="x")
        
        self.lookup_button = ctk.CTkButton(entry_container, text="QUERY ENGINE", height=40, width=150, 
                                           fg_color=COLORS["primary"], hover_color=COLORS["accent"],
                                           font=ctk.CTkFont(size=13, weight="bold"),
                                           command=self.quick_analysis)
        self.lookup_button.pack(side="right")

    def quick_analysis(self):
        ticker = self.lookup_entry.get().upper()
        if ticker:
            self.show_analysis(ticker)

    def create_stat_card(self, parent, title, value, column):
        card = ctk.CTkFrame(parent, corner_radius=12, fg_color=COLORS["surface"], border_width=1, border_color="#1E293B")
        card.grid(row=0, column=column, padx=8, pady=10, sticky="nsew")
        parent.grid_columnconfigure(column, weight=1)
        
        ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=11, weight="bold"), text_color=COLORS["text_dark"]).pack(pady=(15, 0))
        ctk.CTkLabel(card, text=value, font=ctk.CTkFont(size=24, weight="bold"), text_color=COLORS["text"]).pack(pady=(5, 15))

    def show_portfolio(self):
        self.show_page("Portfolio")
        for widget in self.pages["Portfolio"].winfo_children():
            widget.destroy()
            
        header = ctk.CTkLabel(self.pages["Portfolio"], text="PORTFOLIO GOVERNANCE", 
                              font=ctk.CTkFont(size=24, weight="bold"), text_color=COLORS["text"])
        header.pack(pady=(20, 10))

        # Trade Execution Form
        form_card = ctk.CTkFrame(self.pages["Portfolio"], fg_color=COLORS["surface"], corner_radius=12, border_width=1, border_color="#1E293B")
        form_card.pack(fill="x", padx=30, pady=10)
        
        ctk.CTkLabel(form_card, text="TRADE EXECUTION DESK", font=ctk.CTkFont(size=12, weight="bold"), text_color=COLORS["text_dark"]).pack(pady=(10, 5))
        
        f_row = ctk.CTkFrame(form_card, fg_color="transparent")
        f_row.pack(fill="x", padx=20, pady=(0, 15))
        
        self.p_ticker = ctk.CTkEntry(f_row, placeholder_text="Ticker", width=120, height=35)
        self.p_ticker.pack(side="left", padx=5)
        self.p_qty = ctk.CTkEntry(f_row, placeholder_text="QTY", width=100, height=35)
        self.p_qty.pack(side="left", padx=5)
        self.p_price = ctk.CTkEntry(f_row, placeholder_text="Avg Price", width=120, height=35)
        self.p_price.pack(side="left", padx=5)
        
        ctk.CTkButton(f_row, text="OPEN POSITION", width=140, height=35, 
                     fg_color=COLORS["primary"], hover_color=COLORS["accent"],
                     font=ctk.CTkFont(size=12, weight="bold"),
                     command=self.add_position).pack(side="left", padx=(15, 5))
        
        ctk.CTkButton(f_row, text="RESET", width=80, height=35, fg_color="#334155", 
                     command=self.clear_portfolio_form).pack(side="right")

        # Holdings table
        table_card = ctk.CTkFrame(self.pages["Portfolio"], fg_color=COLORS["surface"], corner_radius=12, border_width=1, border_color="#1E293B")
        table_card.pack(fill="both", expand=True, padx=30, pady=10)
        
        table_frame = ctk.CTkScrollableFrame(table_card, fg_color="transparent", label_text="ACTIVE ASSET HOLDINGS", label_text_color=COLORS["text_dark"])
        table_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        summary = self.portfolio.get_summary()
        if summary['holdings']:
            headers = ["Ticker", "Shares", "Avg Cost", "Current", "Value", "ROI %", "Actions"]
            for i, h in enumerate(headers):
                ctk.CTkLabel(table_frame, text=h, font=ctk.CTkFont(weight="bold")).grid(row=0, column=i, padx=10, pady=5)
            
            for r, h in enumerate(summary['holdings'], 1):
                ticker = h['ticker']
                ctk.CTkLabel(table_frame, text=ticker).grid(row=r, column=0, padx=10, pady=2)
                ctk.CTkLabel(table_frame, text=str(h['quantity'])).grid(row=r, column=1, padx=10, pady=2)
                ctk.CTkLabel(table_frame, text=f"${h['purchase_price']:.2f}").grid(row=r, column=2, padx=10, pady=2)
                ctk.CTkLabel(table_frame, text=f"${h['current_price']:.2f}").grid(row=r, column=3, padx=10, pady=2)
                ctk.CTkLabel(table_frame, text=f"${h['current_value']:.2f}").grid(row=r, column=4, padx=10, pady=2)
                color = "green" if h['profit_loss_pct'] >= 0 else "red"
                ctk.CTkLabel(table_frame, text=f"{h['profit_loss_pct']:.2f}%", text_color=color).grid(row=r, column=5, padx=10, pady=2)
                
                # Action buttons
                actions_frame = ctk.CTkFrame(table_frame, fg_color="transparent")
                actions_frame.grid(row=r, column=6, padx=10, pady=2)
                
                adj_btn = ctk.CTkButton(actions_frame, text="✏️", width=30, height=25, 
                                       command=lambda t=ticker, q=h['quantity'], p=h['purchase_price']: self.adjust_position(t, q, p))
                adj_btn.pack(side="left", padx=2)
                
                rem_btn = ctk.CTkButton(actions_frame, text="🗑️", width=30, height=25, fg_color="red", hover_color="darkred",
                                       command=lambda t=ticker: self.remove_position(t))
                rem_btn.pack(side="left", padx=2)
        else:
            ctk.CTkLabel(table_frame, text="Portfolio is empty").pack(pady=20)

    def add_position(self):
        ticker = self.p_ticker.get().upper()
        try:
            qty = float(self.p_qty.get())
            price = float(self.p_price.get())
            if ticker and qty > 0 and price > 0:
                if self.portfolio.add_stock(ticker, qty, price):
                    self.show_portfolio()
                    self.status_label.configure(text=f"Updated {ticker} in portfolio", text_color="green")
        except Exception as e:
            self.status_label.configure(text=f"Error adding position: {str(e)}", text_color="red")

    def adjust_position(self, ticker, qty, price):
        self.p_ticker.delete(0, "end")
        self.p_ticker.insert(0, ticker)
        self.p_qty.delete(0, "end")
        self.p_qty.insert(0, str(qty))
        self.p_price.delete(0, "end")
        self.p_price.insert(0, str(price))

    def remove_position(self, ticker):
        if self.portfolio.remove_stock(ticker):
            self.show_portfolio()
            self.status_label.configure(text=f"Removed {ticker} from portfolio", text_color="orange")

    def clear_portfolio_form(self):
        self.p_ticker.delete(0, "end")
        self.p_qty.delete(0, "end")
        self.p_price.delete(0, "end")

    def show_analysis(self, ticker=None):
        self.show_page("Analysis")
        for widget in self.pages["Analysis"].winfo_children():
            widget.destroy()
            
        header = ctk.CTkLabel(self.pages["Analysis"], text="QUANTITATIVE TECHNICALS", 
                              font=ctk.CTkFont(size=24, weight="bold"), text_color=COLORS["text"])
        header.pack(pady=(20, 10))
        
        search_card = ctk.CTkFrame(self.pages["Analysis"], fg_color=COLORS["surface"], corner_radius=12, border_width=1, border_color="#1E293B")
        search_card.pack(fill="x", padx=30, pady=10)
        
        search_row = ctk.CTkFrame(search_card, fg_color="transparent")
        search_row.pack(fill="x", padx=20, pady=15)
        
        self.a_ticker = ctk.CTkEntry(search_row, placeholder_text="Enter Operational Ticker...", height=40)
        if ticker: self.a_ticker.insert(0, ticker)
        self.a_ticker.pack(side="left", padx=(0, 15), expand=True, fill="x")
        
        self.a_button = ctk.CTkButton(search_row, text="START ENGINE", height=40, width=150,
                                     fg_color=COLORS["primary"], hover_color=COLORS["accent"],
                                     font=ctk.CTkFont(size=13, weight="bold"),
                                     command=self.run_analysis)
        self.a_button.pack(side="right")
        
        self.chart_container = ctk.CTkFrame(self.pages["Analysis"], fg_color=COLORS["surface"], corner_radius=12, border_width=1, border_color="#1E293B")
        self.chart_container.pack(fill="both", expand=True, padx=30, pady=10)
        
        if ticker:
            self.run_analysis()

    def run_analysis(self):
        ticker = self.a_ticker.get().upper()
        if not ticker: return
        
        # Clear container
        for widget in self.chart_container.winfo_children():
            widget.destroy()
            
        loading = ctk.CTkLabel(self.chart_container, text="Fetching Market Data...")
        loading.pack(expand=True)
        
        def fetch_data():
            data = self.fetcher.get_historical_data(ticker, period="1y")
            self.after(0, lambda: self.render_chart(data, ticker))
            
        threading.Thread(target=fetch_data, daemon=True).start()

    def render_chart(self, data, ticker):
        for widget in self.chart_container.winfo_children():
            widget.destroy()
            
        if data is not None and not data.empty:
            GUICharts.create_stock_chart(self.chart_container, data, ticker)
        else:
            ctk.CTkLabel(self.chart_container, text=f"Data for {ticker} not found.").pack(expand=True)

    def show_predictions(self):
        self.show_page("AI Predictions")
        for widget in self.pages["AI Predictions"].winfo_children(): widget.destroy()
        
        header = ctk.CTkLabel(self.pages["AI Predictions"], text="PREDICTIVE INTELLIGENCE", 
                             font=ctk.CTkFont(size=24, weight="bold"), text_color=COLORS["text"])
        header.pack(pady=(20, 10))
        
        # Prediction Input Card
        input_card = ctk.CTkFrame(self.pages["AI Predictions"], fg_color=COLORS["surface"], corner_radius=12, border_width=1, border_color="#1E293B")
        input_card.pack(fill="x", padx=30, pady=10)
        
        col_frame = ctk.CTkFrame(input_card, fg_color="transparent")
        col_frame.pack(fill="x", padx=20, pady=15)
        
        self.pred_ticker = ctk.CTkEntry(col_frame, placeholder_text="Symbol Target", height=40, width=200)
        self.pred_ticker.pack(side="left", padx=(0, 15))
        
        self.fast_mode_var = tk.BooleanVar(value=True)
        self.fast_mode_cb = ctk.CTkCheckBox(col_frame, text="Fast Analysis Mode", text_color=COLORS["text_dark"], 
                                            variable=self.fast_mode_var, font=ctk.CTkFont(size=12))
        self.fast_mode_cb.pack(side="left", padx=10)
        
        self.pred_btn = ctk.CTkButton(col_frame, text="EXECUTE FORECAST", height=40, width=180,
                                     fg_color=COLORS["primary"], hover_color=COLORS["accent"],
                                     font=ctk.CTkFont(size=13, weight="bold"),
                                     command=self.run_prediction)
        self.pred_btn.pack(side="right")
        
        # Results area splits into Table and Insights
        self.results_container = ctk.CTkFrame(self.pages["AI Predictions"], fg_color="transparent")
        self.results_container.pack(fill="both", expand=True, padx=30, pady=10)
        
        self.pred_results = ctk.CTkFrame(self.results_container, fg_color=COLORS["surface"], corner_radius=12, border_width=1, border_color="#1E293B")
        self.pred_results.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        self.pred_insights = ctk.CTkScrollableFrame(self.results_container, fg_color=COLORS["surface"], corner_radius=12, 
                                                   border_width=1, border_color="#1E293B", label_text="INTELLIGENCE FEED")
        self.pred_insights.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        
        self.results_container.grid_columnconfigure(0, weight=3)
        self.results_container.grid_columnconfigure(1, weight=2)
        self.results_container.grid_rowconfigure(0, weight=1)

    def run_prediction(self):
        ticker = self.pred_ticker.get().upper()
        if not ticker: return
        
        for widget in self.pred_results.winfo_children(): widget.destroy()
        for widget in self.pred_insights.winfo_children(): widget.destroy()
        
        loading = ctk.CTkLabel(self.pred_results, text="SYNCHRONIZING QUANTUM ENGINES...", font=ctk.CTkFont(size=14, weight="bold"), text_color=COLORS["accent"])
        loading.pack(expand=True)
        
        def job():
            is_fast = self.fast_mode_var.get()
            res = self.predictor.predict_ensemble(ticker, days=7) # Unified 7-day focus
            self.after(0, lambda: self.render_predictions(res, ticker))
        threading.Thread(target=job, daemon=True).start()

    def render_predictions(self, res, ticker):
        for widget in self.pred_results.winfo_children(): widget.destroy()
        
        if res and res['success']:
            ctk.CTkLabel(self.pred_results, text=f"FORECAST HORIZON: {ticker}", 
                         font=ctk.CTkFont(size=18, weight="bold"), text_color=COLORS["text"]).pack(pady=15)
            
            # Prediction Results Snapshot
            scroll_f = ctk.CTkScrollableFrame(self.pred_results, fg_color="transparent", height=250)
            scroll_f.pack(fill="both", expand=True, padx=20)
            
            for p in res['forecast']:
                row = ctk.CTkFrame(scroll_f, fg_color="transparent")
                row.pack(fill="x", pady=2)
                ctk.CTkLabel(row, text=p['date'], font=ctk.CTkFont(family="Courier", size=13), text_color=COLORS["text_dark"]).pack(side="left")
                ctk.CTkLabel(row, text=f"${p['price']:.2f}", font=ctk.CTkFont(family="Courier", size=14, weight="bold"), text_color=COLORS["text"]).pack(side="right")
            
            # Action Button
            dash_btn = ctk.CTkButton(self.pred_results, text="OPEN INTERACTIVE HUB", 
                                    height=45, fg_color=COLORS["primary"], hover_color=COLORS["accent"],
                                    font=ctk.CTkFont(size=13, weight="bold"),
                                    command=lambda: self.open_plotly_dashboard(res))
            dash_btn.pack(pady=20, padx=40, fill="x")
            
            # Insights
            for insight in res['insights']:
                ctk.CTkLabel(self.pred_insights, text=f"• {insight}", anchor="w", wraplength=350, 
                             font=ctk.CTkFont(size=12), text_color=COLORS["text"]).pack(fill="x", padx=10, pady=5)
        else:
            ctk.CTkLabel(self.pred_results, text=f"ERROR: {res.get('error', 'Pulse Interrupted')}", text_color=COLORS["danger"]).pack(expand=True)

    def open_plotly_dashboard(self, res):
        path = PlotlyDashboard.create_forecast_dashboard(res)
        webbrowser.open(f"file://{path}")

    def show_sentiment(self):
        self.show_page("Sentiment")
        for widget in self.pages["Sentiment"].winfo_children(): widget.destroy()
        
        header = ctk.CTkLabel(self.pages["Sentiment"], text="MARKET SENTIMENT CORE", 
                             font=ctk.CTkFont(size=24, weight="bold"), text_color=COLORS["text"])
        header.pack(pady=(20, 10))
        
        search_card = ctk.CTkFrame(self.pages["Sentiment"], fg_color=COLORS["surface"], corner_radius=12, border_width=1, border_color="#1E293B")
        search_card.pack(fill="x", padx=30, pady=10)
        
        s_row = ctk.CTkFrame(search_card, fg_color="transparent")
        s_row.pack(fill="x", padx=20, pady=15)
        
        self.sent_ticker = ctk.CTkEntry(s_row, placeholder_text="Symbol Analysis Target", height=40)
        self.sent_ticker.pack(side="left", padx=(0, 15), expand=True, fill="x")
        
        self.sent_btn = ctk.CTkButton(s_row, text="SCAN GLOBAL NEWS", height=40, width=180,
                                     fg_color=COLORS["primary"], hover_color=COLORS["accent"],
                                     font=ctk.CTkFont(size=12, weight="bold"),
                                     command=self.run_sentiment)
        self.sent_btn.pack(side="right")
        
        self.sent_results = ctk.CTkScrollableFrame(self.pages["Sentiment"], fg_color=COLORS["surface"], corner_radius=12, 
                                                   border_width=1, border_color="#1E293B", label_text="INTELLIGENCE STREAM")
        self.sent_results.pack(fill="both", expand=True, padx=30, pady=10)

    def run_sentiment(self):
        ticker = self.sent_ticker.get().upper()
        if not ticker: return
        
        for widget in self.sent_results.winfo_children(): widget.destroy()
        loading = ctk.CTkLabel(self.sent_results, text="SCANNING GLOBAL NEWS REPOSITORIES...", font=ctk.CTkFont(size=13, weight="bold"), text_color=COLORS["accent"])
        loading.pack(expand=True)
        
        def job():
            res = self.sentiment_analyzer.analyze_stock(ticker)
            self.after(0, lambda: self.render_sentiment(res))
        threading.Thread(target=job, daemon=True).start()

    def render_sentiment(self, res):
        for widget in self.sent_results.winfo_children(): widget.destroy()
        if res and res['num_articles'] > 0:
            color = COLORS["success"] if res['sentiment_score'] > 0 else COLORS["danger"] if res['sentiment_score'] < 0 else COLORS["text_dark"]
            ctk.CTkLabel(self.sent_results, text=f"AGGREGATE BIAS: {res['sentiment_label'].upper()} ({res['sentiment_score']:.2f})", 
                        text_color=color, font=ctk.CTkFont(size=18, weight="bold")).pack(pady=15)
            
            for art in res['articles']:
                item = ctk.CTkFrame(self.sent_results, fg_color="transparent")
                item.pack(fill="x", padx=10, pady=8)
                ctk.CTkLabel(item, text=art['title'], font=ctk.CTkFont(size=13, weight="bold"), text_color=COLORS["text"], wraplength=600, anchor="w").pack(fill="x")
                ctk.CTkLabel(item, text=f"{art['source']} • Bias: {art['sentiment_label']}", font=ctk.CTkFont(size=11), text_color=COLORS["text_dark"], anchor="w").pack(fill="x")
                ctk.CTkFrame(self.sent_results, height=1, fg_color="#1E293B").pack(fill="x", padx=10)
        else:
            ctk.CTkLabel(self.sent_results, text="No intelligence found for this symbol.", text_color=COLORS["text_dark"]).pack(expand=True)

    def show_alerts(self):
        self.show_page("Alerts")
        for widget in self.pages["Alerts"].winfo_children(): widget.destroy()
        header = ctk.CTkLabel(self.pages["Alerts"], text="RISK & PRICE OVERLAYS", 
                             font=ctk.CTkFont(size=24, weight="bold"), text_color=COLORS["text"])
        header.pack(pady=(20, 10))
        
        card = ctk.CTkFrame(self.pages["Alerts"], fg_color=COLORS["surface"], corner_radius=12, border_width=1, border_color="#1E293B")
        card.pack(fill="both", expand=True, padx=40, pady=40)
        
        ctk.CTkLabel(card, text="🔔 OPERATIONAL MONITORING ACTIVE", font=ctk.CTkFont(size=16, weight="bold"), text_color=COLORS["success"]).pack(expand=True)
        ctk.CTkLabel(card, text="Systems are scanning portfolio positions for critical volatility spikes and target triggers.", font=ctk.CTkFont(size=12), text_color=COLORS["text_dark"]).pack(pady=(0, 40))

    def show_backtesting(self):
        self.show_page("Backtesting")
        # Implementation skeleton
        ctk.CTkLabel(self.pages["Backtesting"], text="Strategy Engine", 
                    font=ctk.CTkFont(size=24, weight="bold")).pack(pady=20)

    def show_settings(self):
        self.show_page("Settings")
        for widget in self.pages["Settings"].winfo_children(): widget.destroy()
        header = ctk.CTkLabel(self.pages["Settings"], text="CORE REPOSITORY SETTINGS", 
                             font=ctk.CTkFont(size=24, weight="bold"), text_color=COLORS["text"])
        header.pack(pady=(20, 10))
        
        status_card = ctk.CTkFrame(self.pages["Settings"], fg_color=COLORS["surface"], corner_radius=12, border_width=1, border_color="#1E293B")
        status_card.pack(fill="x", padx=40, pady=20)
        
        ctk.CTkLabel(status_card, text="DIAGNOSTIC ENGINE STATUS", font=ctk.CTkFont(size=12, weight="bold"), text_color=COLORS["text_dark"]).pack(pady=15)
        
        lines = [
            ("Market Data Gateway", "yfinance: Operational"),
            ("News Intelligence", f"NewsAPI: {'Active' if config.has_news_api_key() else 'Key Required'}"),
            ("Notification Node", f"SMTP Relay: {'Active' if config.has_email_config() else 'Not Wired'}")
        ]
        
        for label, val in lines:
            row = ctk.CTkFrame(status_card, fg_color="transparent")
            row.pack(fill="x", padx=30, pady=4)
            ctk.CTkLabel(row, text=label, font=ctk.CTkFont(size=13), text_color=COLORS["text"]).pack(side="left")
            status_color = COLORS["success"] if "Active" in val or "Operational" in val else COLORS["danger"]
            ctk.CTkLabel(row, text=val, font=ctk.CTkFont(size=13, weight="bold"), text_color=status_color).pack(side="right")
        
        # Performance Card
        perf_card = ctk.CTkFrame(self.pages["Settings"], fg_color=COLORS["surface"], corner_radius=12, border_width=1, border_color="#1E293B")
        perf_card.pack(fill="x", padx=40, pady=10)
        
        from src.models.lstm_predictor import TENSORFLOW_AVAILABLE
        tf_status = "ACTIVE (DL)" if TENSORFLOW_AVAILABLE else "LIGHT (STATS)"
        
        ctk.CTkLabel(perf_card, text=f"ENGINE MODE: {tf_status}", font=ctk.CTkFont(size=13, weight="bold"), text_color=COLORS["primary"]).pack(pady=15)
        ctk.CTkLabel(perf_card, text="Accelerator: FP16 Mixed Precision • Optimized for Minimal Latency", font=ctk.CTkFont(size=11), text_color=COLORS["text_dark"]).pack(pady=(0, 15))

    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)

if __name__ == "__main__":
    app = StockTrackerApp()
    app.mainloop()
