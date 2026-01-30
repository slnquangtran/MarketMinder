"""GUI specific charting utility for the desktop app."""
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from typing import Optional
import tkinter as tk
from src.analysis.technical_indicators import TechnicalIndicators

# Institutional Palette Match
THEME_COLORS = {
    "bg": "#0A0A0F",
    "surface": "#16161E",
    "primary": "#0066FF",
    "accent": "#3B82F6",
    "success": "#00C853",
    "danger": "#FF3D00",
    "text": "#E1E1E6",
    "grid": "#1E293B"
}

class GUICharts:
    """Helper class to embed Matplotlib charts into Tkinter/CustomTkinter."""
    
    @staticmethod
    def create_stock_chart(canvas_parent: tk.Widget, data: pd.DataFrame, ticker: str):
        """Create a technical analysis chart using Matplotlib."""
        # Add indicators
        data = TechnicalIndicators.add_all_indicators(data)
        
        # Create figure with 2:1 ratio for metrics
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        plt.subplots_adjust(hspace=0.08, left=0.08, right=0.95, top=0.92, bottom=0.1)
        
        # Global Style Overhaul
        plt.style.use('dark_background')
        fig.patch.set_facecolor(THEME_COLORS["bg"])
        ax1.set_facecolor(THEME_COLORS["surface"])
        ax2.set_facecolor(THEME_COLORS["surface"])
        
        # Upper Chart: Price & Momentum
        ax1.plot(data.index, data['Close'], label='MARKET PRICE', color=THEME_COLORS["text"], linewidth=1.8, alpha=0.9)
        ax1.plot(data.index, data['SMA_20'], label='SMA-20', color=THEME_COLORS["primary"], linewidth=1.2)
        ax1.plot(data.index, data['SMA_50'], label='SMA-50', color="#FACC15", linewidth=1.2, alpha=0.8)
        
        # Bollinger Bands with clean shading
        ax1.fill_between(data.index, data['BB_Upper'], data['BB_Lower'], color=THEME_COLORS["primary"], alpha=0.12, label='BB Bands')
        
        ax1.set_title(f"TECHNICAL CORE: {ticker}", color=THEME_COLORS["text"], fontsize=13, fontweight='bold', pad=15)
        legend = ax1.legend(loc='upper left', fontsize=9, frameon=False)
        plt.setp(legend.get_texts(), color=THEME_COLORS["text"])
        
        ax1.tick_params(colors=THEME_COLORS["text"], labelsize=9)
        ax1.grid(True, color=THEME_COLORS["grid"], alpha=0.4, linestyle='--')
        
        # Lower Chart: RSI Momentum
        ax2.plot(data.index, data['RSI'], color="#A855F7", linewidth=1.5, label='RSI-14')
        ax2.axhline(70, color=THEME_COLORS["danger"], linestyle='--', alpha=0.6, linewidth=0.8)
        ax2.axhline(30, color=THEME_COLORS["success"], linestyle='--', alpha=0.6, linewidth=0.8)
        
        ax2.fill_between(data.index, 30, 70, color="#A855F7", alpha=0.05)
        ax2.set_ylabel('MOMENTUM', color=THEME_COLORS["text"], fontsize=9, fontweight='bold')
        ax2.set_ylim(0, 100)
        ax2.tick_params(colors=THEME_COLORS["text"], labelsize=8)
        ax2.grid(True, color=THEME_COLORS["grid"], alpha=0.3, linestyle=':')
        
        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=canvas_parent)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill="both", expand=True, padx=5, pady=5)
        canvas.draw()
        
        return canvas
