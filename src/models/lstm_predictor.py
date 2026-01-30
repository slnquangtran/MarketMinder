"""LSTM-based stock price prediction model."""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional, BatchNormalization, Conv1D, GlobalMaxPooling1D
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    
    # Enable Mixed Precision for Hardware Acceleration
    try:
        from tensorflow.keras import mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print(f"TensorFlow Mixed Precision set to: {policy.name}")
    except:
        print("Mixed Precision not supported, using default float32")
        
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. LSTM predictions will be disabled.")

# Decorator fallback for environments without TensorFlow
def tf_function(func):
    if TENSORFLOW_AVAILABLE:
        return tf.function(func)
    return func

from src.data.stock_fetcher import StockFetcher
from src.analysis.technical_indicators import TechnicalIndicators
from src.models.sentiment_analyzer import SentimentAnalyzer

class LSTMPredictor:
    """Enhanced Bidirectional LSTM model for stock price prediction."""
    
    def __init__(self, sequence_length: int = 60, model_dir: str = "models/saved"):
        """Initialize LSTM predictor.
        
        Args:
            sequence_length: Number of days to use for prediction
            model_dir: Directory to save/load models
        """
        self.sequence_length = sequence_length
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.scaler = None
        self.fetcher = StockFetcher()
        self.sentiment_analyzer = SentimentAnalyzer()
        self._pred_cache = {} # Cache for final predictions: {ticker: (ts, results)}
    
    def _add_indicators(self, data: pd.DataFrame, ticker: str = None, 
                       include_sentiment: bool = True) -> pd.DataFrame:
        """Add ONLY required indicators for speed."""
        data = data.copy()
        # SMA_20
        data['SMA_20'] = TechnicalIndicators.sma(data, 20)
        # RSI
        data['RSI'] = TechnicalIndicators.rsi(data)
        # MACD
        macd_data = TechnicalIndicators.macd(data)
        data['MACD'] = macd_data['MACD']
        data['MACD_Hist'] = macd_data['Histogram']
        # ATR
        data['ATR'] = TechnicalIndicators.atr(data)
        
        data['Sentiment'] = 0.0
        if include_sentiment and ticker and self.sentiment_analyzer.is_configured():
            try:
                res = self.sentiment_analyzer.analyze_stock(ticker)
                data['Sentiment'] = res['sentiment_score']
            except:
                pass
        return data

    def prepare_data(self, data: pd.DataFrame, ticker: str = None,
                    features: List[str] = None, include_sentiment: bool = True) -> Tuple[np.ndarray, np.ndarray, object]:
        """Prepare data for LSTM training with optimized indicator calculation."""
        if not TENSORFLOW_AVAILABLE:
            return None, None, None
        
        # Default advanced features
        if features is None:
            features = ['Close', 'Volume', 'SMA_20', 'RSI', 'MACD', 'MACD_Hist', 'ATR', 'Sentiment']
        
        # Add required indicators
        data = self._add_indicators(data, ticker, include_sentiment)

        # Ensure all requested features exist
        available_features = [f for f in features if f in data.columns]
        
        # Select and clean features
        feature_data = data[available_features].dropna()
        
        if len(feature_data) < self.sequence_length:
            return None, None, None

        # Scale data
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(feature_data)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            y.append(scaled_data[i, 0])  # Predict Close price
        
        return np.array(X), np.array(y), scaler
    
    def build_model(self, input_shape: Tuple) -> Sequential:
        """Build High-Performance Hybrid Conv1D + GRU architecture."""
        if not TENSORFLOW_AVAILABLE:
            return None
        
        model = Sequential([
            # Conv1D for local pattern extraction (faster than recurrent layers)
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            BatchNormalization(),
            
            # GRU for temporal dependencies (faster than LSTM)
            GRU(128, return_sequences=True),
            Dropout(0.3),
            
            GRU(64),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            # Force output to float32 for mixed precision stability
            Dense(1, dtype='float32') 
        ])
        
        # Use a high-performance optimizer
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse')
        return model
    
    def train(self, ticker: str, epochs: int = 40, batch_size: int = 64,
             validation_split: float = 0.2) -> Dict[str, Any]:
        """Train model using high-performance tf.data pipeline."""
        if not TENSORFLOW_AVAILABLE:
            return {'error': 'TensorFlow not available'}
        
        data = self.fetcher.get_historical_data(ticker, period='2y')
        if data is None or len(data) < self.sequence_length + 20:
            return {'error': 'Insufficient data'}
        
        X, y, scaler = self.prepare_data(data, ticker=ticker)
        if X is None: return {'error': 'Prep failed'}
        self.scaler = scaler
        
        # ⚡ Ultra-Fast tf.data Pipeline
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.shuffle(len(X)).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
        
        val_size = int(len(X) * validation_split)
        train_ds = dataset.skip(val_size // batch_size)
        val_ds = dataset.take(val_size // batch_size)

        self.model = self.build_model((X.shape[1], X.shape[2]))
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
            # Faster learning rate adjustment
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
        ]
        
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=0
        )
        
        self.save_model(ticker)
        return {'success': True, 'ticker': ticker}
    
    @tf_function
    def _compiled_predict(self, X_pred):
        """Accelerated static graph inference."""
        return self.model(X_pred, training=False)

    def predict(self, ticker: str, days: int = 7, fast_mode: bool = False) -> Optional[List[Dict]]:
        """Predict future stock prices with sub-100ms graph inference and caching."""
        if not TENSORFLOW_AVAILABLE:
            return None
        
        # Check prediction cache (30 min)
        now = pd.Timestamp.now().timestamp()
        if ticker in self._pred_cache:
            ts, cached_res = self._pred_cache[ticker]
            if now - ts < 1800: # 30 mins
                return cached_res
        
        if self.model is None:
            if not self.load_model(ticker):
                print(f"Auto-training model for {ticker}...")
                self.train(ticker, epochs=15 if fast_mode else 30)
        
        period = '6mo' if fast_mode else '1y'
        data = self.fetcher.get_historical_data(ticker, period=period)
        if data is None or len(data) < self.sequence_length + 20: return None
        
        data = self._add_indicators(data, ticker, include_sentiment=not fast_mode)
        features = ['Close', 'Volume', 'SMA_20', 'RSI', 'MACD', 'MACD_Hist', 'ATR', 'Sentiment']
        available_features = [f for f in features if f in data.columns]
        feature_data = data[available_features].dropna()
        
        if len(feature_data) < self.sequence_length or self.scaler is None: return None
            
        # Feature validation
        if hasattr(self.scaler, 'n_features_in_') and self.scaler.n_features_in_ != len(available_features):
            self.model = None
            return self.predict(ticker, days, fast_mode)
            
        scaled_data = self.scaler.transform(feature_data)
        last_sequence = scaled_data[-self.sequence_length:]
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for i in range(days):
            X_pred = current_sequence.reshape(1, self.sequence_length, len(available_features))
            # ⚡ Execute compiled static graph inference
            raw_pred = self._compiled_predict(tf.convert_to_tensor(X_pred, dtype=tf.float32)).numpy()[0, 0]
            
            # Stable results: Apply a decay-weighted smoothing (blend with previous price)
            last_price_scaled = current_sequence[-1, 0]
            smoothing_factor = 0.3 # 30% new pred, 70% historical momentum
            pred_scaled = (raw_pred * smoothing_factor) + (last_price_scaled * (1 - smoothing_factor))
            
            dummy = np.zeros((1, len(available_features)))
            dummy[0, 0] = pred_scaled
            pred_price = self.scaler.inverse_transform(dummy)[0, 0]
            
            # Guarantee non-negative prices
            pred_price = max(0.01, float(pred_price))
            
            predictions.append({
                'date': (pd.Timestamp.now() + pd.Timedelta(days=i+1)).strftime('%Y-%m-%d'),
                'predicted_price': pred_price
            })
            
            new_row = current_sequence[-1].copy()
            new_row[0] = pred_scaled
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        # Update cache
        self._pred_cache[ticker] = (now, predictions)
        return predictions
    
    def save_model(self, ticker: str):
        """Save model and scaler to disk.
        
        Args:
            ticker: Stock ticker symbol
        """
        if not TENSORFLOW_AVAILABLE or self.model is None:
            return
        
        model_path = self.model_dir / f"{ticker}_lstm.h5"
        scaler_path = self.model_dir / f"{ticker}_scaler.pkl"
        
        self.model.save(model_path)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load_model(self, ticker: str) -> bool:
        """Load model and scaler from disk.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            True if successful, False otherwise
        """
        if not TENSORFLOW_AVAILABLE:
            return False
        
        model_path = self.model_dir / f"{ticker}_lstm.h5"
        scaler_path = self.model_dir / f"{ticker}_scaler.pkl"
        
        if not model_path.exists() or not scaler_path.exists():
            return False
        
        try:
            # Fix deserialization issues (common in mixed-precision or differing Keras versions)
            custom_objects = {'mse': 'mse'}
            self.model = keras.models.load_model(model_path, custom_objects=custom_objects)
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            # If load fails, delete corrupted files to force retraining
            try:
                model_path.unlink(missing_ok=True)
                scaler_path.unlink(missing_ok=True)
            except:
                pass
            return False
