# /Users/shusheng/PythonProjects/alpha_lab/alpha_lab/data/sources/__init__.py

"""
Data sources: pluggable readers for different formats.

Current: CSV (wide/long format)
Future: Parquet, HDF5, SQL, API
"""

from alpha_lab.data.sources.csv_source import load_csv_prices

__all__ = ["load_csv_prices"]
from alpha_lab.data.sources.base import DataSource
from alpha_lab.data.sources.parquet_source import ParquetDataSource

__all__ = ["DataSource", "ParquetDataSource"]
