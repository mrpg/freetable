"""freetable: Convert statsmodels results to publication-ready LaTeX tables.

This package provides few, highly opinionated functions to create beautiful
LaTeX tables from statsmodels regression results.

Requirements:
    - Python package: statsmodels
    - LaTeX packages: threeparttable, siunitx, booktabs
"""

from freetable.table import table

__version__ = "0.0.1"
__author__ = "Max R. P. Grossmann"
__all__ = ["table"]
