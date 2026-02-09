"""
politext - Political Text Data Collection System for Sentiment Bias Research

A modular Python package for collecting, processing, and organizing political
text data for academic sentiment bias research.
"""

__version__ = "0.1.0"
__author__ = "Neil Khandelwal"

from politext.config import Config, load_config

__all__ = [
    "__version__",
    "Config",
    "load_config",
]
