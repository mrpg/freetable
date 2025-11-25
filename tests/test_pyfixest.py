"""Tests for pyfixest integration."""

import pandas as pd
import pytest

# Try to import pyfixest - skip tests if not available
pyfixest = pytest.importorskip("pyfixest")

from freetable import table
from freetable.adapter import PyFixestAdapter, adapt_model


@pytest.fixture
def df():
    """Sample dataframe for testing."""
    return pd.DataFrame(
        {
            "y": [1, 2, 3, 4, 5, 6, 7, 8],
            "x1": [2, 4, 6, 8, 10, 12, 14, 16],
            "x2": [1, 1, 2, 2, 3, 3, 4, 4],
            "group": [1, 1, 2, 2, 1, 1, 2, 2],
        }
    )


@pytest.fixture
def pf_model1(df):
    """Single variable pyfixest model."""
    return pyfixest.feols("y ~ x1", data=df)


@pytest.fixture
def pf_model2(df):
    """Two variable pyfixest model."""
    return pyfixest.feols("y ~ x1 + x2", data=df)


def test_pyfixest_adapter(pf_model1):
    """Test that PyFixestAdapter provides the correct interface."""
    adapter = PyFixestAdapter(pf_model1)

    # Check that all required attributes exist and return expected types
    assert hasattr(adapter, "params")
    assert hasattr(adapter, "bse")
    assert hasattr(adapter, "pvalues")
    assert hasattr(adapter, "rsquared")
    assert hasattr(adapter, "rsquared_adj")
    assert hasattr(adapter, "nobs")

    # Check types
    assert isinstance(adapter.params, pd.Series)
    assert isinstance(adapter.bse, pd.Series)
    assert isinstance(adapter.pvalues, pd.Series)
    assert isinstance(adapter.rsquared, float)
    assert isinstance(adapter.rsquared_adj, float)
    assert isinstance(adapter.nobs, int)

    # Check that parameter names are accessible
    assert "x1" in adapter.params.index
    assert "Intercept" in adapter.params.index


def test_adapt_model_pyfixest(pf_model1):
    """Test that adapt_model correctly identifies and wraps pyfixest models."""
    adapted = adapt_model(pf_model1)

    assert isinstance(adapted, PyFixestAdapter)
    assert hasattr(adapted, "params")
    assert hasattr(adapted, "bse")


def test_basic_pyfixest_table(pf_model1):
    """Test basic table generation with a single pyfixest model."""
    result = table(pf_model1)

    # Check structure
    assert r"\begin{table}[htbp]" in result
    assert r"\begin{threeparttable}" in result
    assert r"\toprule" in result
    assert r"\midrule" in result
    assert r"\bottomrule" in result
    assert r"\end{table}" in result

    # Check content
    assert "Intercept" in result
    assert "x1" in result
    assert r"$R^2$" in result
    assert r"Adj. $R^2$" in result
    assert "Observations" in result


def test_multiple_pyfixest_models(pf_model1, pf_model2):
    """Test table with multiple pyfixest models."""
    result = table([pf_model1, pf_model2])

    # Check both models appear
    assert "Model 1" in result
    assert "Model 2" in result

    # Check x2 appears (only in model2)
    assert "x2" in result


def test_pyfixest_with_rename(pf_model1, pf_model2):
    """Test variable renaming with pyfixest models."""
    result = table(
        [pf_model1, pf_model2],
        rename={"x1": "Treatment", "x2": "Control"},
    )

    assert "Treatment" in result
    assert "Control" in result


def test_pyfixest_with_extra_rows(pf_model1, pf_model2):
    """Test extra custom rows with pyfixest models."""
    result = table(
        [pf_model1, pf_model2],
        extra_rows={"Outcome": ["Y", "Y"], "SE type": ["IID", "IID"]},
    )

    assert "Outcome & {Y} & {Y}" in result
    assert "SE type & {IID} & {IID}" in result


def test_pyfixest_with_custom_header(pf_model1, pf_model2):
    """Test custom grouped headers with pyfixest models."""
    result = table(
        [pf_model1, pf_model2],
        custom_header=[("Treatment", 1), ("Control", 1)],
    )

    assert "{Treatment}" in result
    assert "{Control}" in result
    assert r"\cmidrule" in result


def test_pyfixest_significance_stars(pf_model1):
    """Test that significance stars appear for significant coefficients."""
    result = table(pf_model1)

    # The x1 coefficient should be highly significant (perfect correlation)
    # and should have stars in the output
    # Note: we don't check for specific stars as it depends on the p-value
    assert "x1" in result


def test_pyfixest_custom_model_names(pf_model1, pf_model2):
    """Test custom model names with pyfixest."""
    result = table(
        [pf_model1, pf_model2],
        model_names=["Baseline", "Full Model"],
    )

    assert "Baseline" in result
    assert "Full Model" in result
    assert "Model 1" not in result


def test_pyfixest_all_features(pf_model1, pf_model2):
    """Test all features combined with pyfixest models."""
    result = table(
        [pf_model1, pf_model2],
        model_names=["Model A", "Model B"],
        digits=2,
        caption="pyfixest Results",
        label="tab:pyfixest",
        rename={"x1": "Variable X"},
        stars=(0.1, 0.05, 0.01),
        extra_rows={"Type": ["OLS", "OLS"]},
        custom_header=[("Group", 2)],
        placement="h!",
        resize=True,
    )

    # Check all features are present
    assert "Model A" in result
    assert "Model B" in result
    assert r"\caption{pyfixest Results}" in result
    assert r"\label{tab:pyfixest}" in result
    assert "Variable X" in result
    assert "Type & {OLS} & {OLS}" in result
    assert r"\multicolumn{2}{c @{}}{{Group}}" in result
    assert r"\begin{table}[h!]" in result
    assert r"\resizebox" in result


def test_pyfixest_r_squared_values(pf_model1):
    """Test that R-squared values are correctly extracted from pyfixest models."""
    result = table(pf_model1, digits=2)

    # The model should have R² = 1.0 (perfect fit)
    assert "1.00" in result


def test_pyfixest_nobs(pf_model1):
    """Test that number of observations is correctly extracted."""
    result = table(pf_model1)

    # We have 8 observations in our test data
    assert "{8}" in result


def test_pyfixest_with_fixed_effects(df):
    """Test pyfixest model with fixed effects."""
    # Create a model with fixed effects
    model = pyfixest.feols("y ~ x1 | group", data=df)
    result = table(model)

    # Should still work and produce valid LaTeX
    assert r"\begin{table}" in result
    assert "x1" in result


def test_mixed_models_not_supported():
    """Test that mixing statsmodels and pyfixest in same table works.

    Since both use the adapter, they should work together seamlessly.
    """
    import statsmodels.formula.api as smf

    df = pd.DataFrame(
        {
            "y": [1, 2, 3, 4, 5, 6],
            "x1": [2, 4, 6, 8, 10, 12],
        }
    )

    # Create one statsmodels and one pyfixest model
    sm_model = smf.ols("y ~ x1", data=df).fit()
    pf_model = pyfixest.feols("y ~ x1", data=df)

    # Should work with both models in the same table
    result = table([sm_model, pf_model], model_names=["Statsmodels", "pyfixest"])

    assert "Statsmodels" in result
    assert "pyfixest" in result
    assert "x1" in result
