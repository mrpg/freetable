"""Adapter for supporting multiple regression libraries.

This module provides a unified interface for accessing regression results
from different libraries (statsmodels, pyfixest, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import pandas as pd


class RegressionResult(Protocol):
    """Protocol defining the interface expected by the table() function.

    This allows the table() function to work with any regression result object
    that provides these attributes.
    """

    @property
    def params(self) -> pd.Series:  # pragma: no cover
        """Coefficient estimates indexed by parameter name."""
        ...

    @property
    def bse(self) -> pd.Series:  # pragma: no cover
        """Standard errors indexed by parameter name."""
        ...

    @property
    def pvalues(self) -> pd.Series:  # pragma: no cover
        """P-values indexed by parameter name."""
        ...

    @property
    def rsquared(self) -> float:  # pragma: no cover
        """R-squared value."""
        ...

    @property
    def rsquared_adj(self) -> float:  # pragma: no cover
        """Adjusted R-squared value."""
        ...

    @property
    def nobs(self) -> int:  # pragma: no cover
        """Number of observations."""
        ...


class PyFixestAdapter:
    """Adapter for pyfixest result objects.

    Wraps a pyfixest Feols object to provide a statsmodels-like interface.
    """

    def __init__(self, model: Any) -> None:
        """Initialize adapter with a pyfixest model.

        Args:
            model: A pyfixest Feols result object.
        """
        self._model = model

    @property
    def params(self) -> pd.Series:
        """Return coefficients as a pandas Series."""
        return self._model.coef()

    @property
    def bse(self) -> pd.Series:
        """Return standard errors as a pandas Series."""
        return self._model.se()

    @property
    def pvalues(self) -> pd.Series:
        """Return p-values as a pandas Series."""
        return self._model.pvalue()

    @property
    def rsquared(self) -> float:
        """Return R-squared value."""
        return self._model._r2

    @property
    def rsquared_adj(self) -> float:
        """Return adjusted R-squared value."""
        return self._model._adj_r2

    @property
    def nobs(self) -> int:
        """Return number of observations."""
        return self._model._N


def adapt_model(model: Any) -> RegressionResult:
    """Adapt a regression model to the standard interface.

    This function detects the type of regression model and wraps it
    with the appropriate adapter if needed. Statsmodels results are
    passed through unchanged as they already have the correct interface.

    Args:
        model: A regression result object from statsmodels, pyfixest, or
              another supported library.

    Returns:
        A model object with the standard RegressionResult interface.

    Raises:
        TypeError: If the model type is not supported.

    Examples:
        >>> import pyfixest as pf
        >>> import pandas as pd
        >>> df = pd.DataFrame({'y': [1, 2, 3], 'x': [4, 5, 6]})
        >>> pf_model = pf.feols("y ~ x", data=df)
        >>> adapted = adapt_model(pf_model)
        >>> adapted.params  # Works like statsmodels
    """
    # Check model type by class name to avoid hard dependency on pyfixest
    model_type = type(model).__name__
    module_name = type(model).__module__

    # pyfixest models
    if "pyfixest" in module_name or model_type == "Feols":
        return PyFixestAdapter(model)

    # statsmodels models - check for the expected interface
    if hasattr(model, "params") and hasattr(model, "bse") and hasattr(model, "pvalues"):
        # Already has the right interface, no adaptation needed
        return model

    # Unsupported model type
    raise TypeError(
        f"Unsupported model type: {model_type} from module {module_name}. "
        "Supported types: statsmodels results, pyfixest Feols."
    )
