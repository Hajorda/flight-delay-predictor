"""Entry point for ``python -m flight_delay.models``.

Runs the full training pipeline: load data → preprocess → build features →
train all models → evaluate → save.
"""

import logging
import runpy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

# Delegate to train.py's __main__ block
runpy.run_module("flight_delay.models.train", run_name="__main__")
