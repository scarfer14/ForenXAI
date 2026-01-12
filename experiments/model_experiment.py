"""
Proxy for experiments: reuse the canonical ForenXAI/src/model_inference.py
to avoid duplication and drift.

It provides a stable, reusable interface for experiments while keeping production inference untouched.
Experiments call the same inference logic without code duplication, can add experiment-only logging or 
timing, avoid messy imports, and allow temporary experimental changes without polluting the core inference code.
"""

import os
import sys

_THIS_DIR = os.path.dirname(__file__)
_SRC_DIR = os.path.abspath(os.path.join(_THIS_DIR, '..', 'src'))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from ForenXAI.experiments.model_experiment import load_models, predict  # type: ignore

__all__ = [
    'load_models',
    'predict',
]
