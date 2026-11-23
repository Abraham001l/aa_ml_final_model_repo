"""Adaptive Route-Recovery Forecasting, a statistical model without fitted weights."""
from .model import Config, forecast_history
from .data import build_history

__all__ = ["Config", "forecast_history", "build_history"]
__version__ = "1.0.0"
