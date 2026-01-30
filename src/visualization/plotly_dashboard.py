"""Module for generating interactive Plotly dashboards."""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List
from pathlib import Path

class PlotlyDashboard:
    """Generates a professional 4-panel interactive dashboard."""
    
    @staticmethod
    def create_forecast_dashboard(res: Dict, output_path: str = "forecast_dashboard.html") -> str:
        """Create a multi-panel Plotly dashboard and save to HTML.
        
        Args:
            res: Results dictionary from EnsemblePredictor
            output_path: Path to save the HTML file
            
        Returns:
            Path to the saved HTML dashboard
        """
        ticker = res['ticker']
        forecast_df = pd.DataFrame(res['forecast'])
        forecast_df['date'] = pd.to_datetime(forecast_df['date'])
        
        hist_prices = pd.Series(res['historical_data'])
        hist_prices.index = pd.to_datetime(hist_prices.index)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f"<b>{ticker} Strategic Forecast</b>", 
                "<b>Model Consensus: LSTM vs ARIMA</b>",
                "<b>Rolling Volatility Pulse</b>",
                "<b>Risk Distribution & VaR Analysis</b>"
            ),
            vertical_spacing=0.15,
            horizontal_spacing=0.1,
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # --- Panel 1: Main Forecast ---
        # Historical
        fig.add_trace(go.Scatter(x=hist_prices.index, y=hist_prices.values, name="Price History", 
                                line=dict(color="#94A3B8", width=1.5), opacity=0.6), row=1, col=1)
        # Forecast
        fig.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['price'], name="Ensemble Core", 
                                line=dict(color="#0066FF", width=4)), row=1, col=1)
        # Confidence Corridor (Glowing)
        fig.add_trace(go.Scatter(
            x=forecast_df['date'].tolist() + forecast_df['date'].tolist()[::-1],
            y=forecast_df['upper'].tolist() + forecast_df['lower'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(0,102,255,0.15)',
            line=dict(color='rgba(0,102,255,0)'),
            hoverinfo="skip",
            name="Confidence Corridor (95%)"
        ), row=1, col=1)
        
        # --- Panel 2: Model Consensus ---
        if forecast_df['lstm'].iloc[0] is not None:
            fig.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['lstm'], name="LSTM Node", 
                                    line=dict(dash='dash', color='#FACC15', width=2)), row=1, col=2)
        if forecast_df['arima'].iloc[0] is not None:
            fig.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['arima'], name="ARIMA Node", 
                                    line=dict(dash='dot', color='#00C853', width=2)), row=1, col=2)
            
        # --- Panel 3: Volatility ---
        vol = hist_prices.pct_change().rolling(20).std() * np.sqrt(252)
        fig.add_trace(go.Scatter(x=vol.index, y=vol.values, name="20D Realized Vol", 
                                line=dict(color="#A855F7", width=2), fill='tozeroy', 
                                fillcolor='rgba(168,85,247,0.1)'), row=2, col=1)
        
        # --- Panel 4: Risk Distribution ---
        risk_metrics = res['risk_metrics']
        errors = np.random.normal(0, risk_metrics['volatility']*10, 1000)
        fig.add_trace(go.Histogram(x=errors, name="Probability density", marker_color="#3B82F6", 
                                  opacity=0.7, nbinsx=40), row=2, col=2)
        
        # Supporting technical overlays
        for level in res['levels']['resistance']:
            fig.add_hline(y=level, line_dash="dash", line_color="#FF3D00", opacity=0.4, 
                         annotation_text="RES", row=1, col=1)
        for level in res['levels']['support']:
            fig.add_hline(y=level, line_dash="dash", line_color="#00C853", opacity=0.4, 
                         annotation_text="SUPP", row=1, col=1)
            
        # Advanced Metric HUD (Annotations)
        fig.add_annotation(
            text=(f"<b>SHARPE:</b> {risk_metrics['sharpe_ratio']:.2f}<br>"
                  f"<b>VOLATILITY:</b> {risk_metrics['volatility']:.2%}<br>"
                  f"<b>VaR (95%):</b> {risk_metrics['var_95']:.2%}"),
            xref="paper", yref="paper",
            x=0.98, y=0.02,
            showarrow=False,
            font=dict(size=12, color="#E1E1E6", family="Courier New"),
            bgcolor="rgba(22,22,30,0.8)",
            bordercolor="#1E293B",
            borderwidth=1,
            align="left"
        )
            
        # Layout enhancements
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0A0A0F",
            plot_bgcolor="rgba(0,0,0,0)",
            title=dict(
                text=f"INSTITUTIONAL INTELLIGENCE HUB: {ticker}",
                font=dict(size=24, family="Arial Black", color="#0066FF"),
                x=0.05, y=0.96
            ),
            height=1000,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, 
                        font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=120, b=80, l=80, r=80)
        )
        
        # X-Axis Styling
        fig.update_xaxes(
            showgrid=True, gridwidth=1, gridcolor="#1E293B",
            showline=True, linewidth=1, linecolor="#1E293B",
            rangeslider_visible=False,
            row=1, col=1
        )
        # Range Selectors
        fig.update_xaxes(
            rangeselector=dict(
                bgcolor="#16161E",
                activecolor="#0066FF",
                font=dict(size=11),
                buttons=list([
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all", label="MAX")
                ])
            ),
            row=1, col=1
        )
        
        # Save and return
        abs_path = str(Path(output_path).absolute())
        fig.write_html(abs_path)
        return abs_path
