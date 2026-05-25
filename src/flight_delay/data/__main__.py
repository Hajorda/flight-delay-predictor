"""Entry point for ``python -m flight_delay.data``.

Generates synthetic flight data for development and testing.
"""

import logging

from flight_delay.data.synthetic import generate_synthetic_data
from flight_delay.utils.config import RAW_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

df = generate_synthetic_data()
print(f"\n✅ Generated {len(df):,} synthetic flight records.")
print(f"   Saved to: {RAW_DIR / 'synthetic_flights.parquet'}")
