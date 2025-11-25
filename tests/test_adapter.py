"""Tests for the adapter module."""

import pandas as pd
import pytest
import statsmodels.formula.api as smf

from freetable.adapter import PyFixestAdapter, adapt_model

# Try to import pyfixest - skip pyfixest tests if not available
pyfixest = pytest.importorskip("pyfixest", minversion=None)


@pytest.fixture
def df():
    """Sample dataframe for testing."""
    return pd.DataFrame(
        {
            "y": [1, 2, 3, 4, 5, 6],
            "x": [2, 4, 6, 8, 10, 12],
        }
    )


@pytest.fixture
def statsmodels_model(df):
    """Statsmodels OLS result."""
    return smf.ols("y ~ x", data=df).fit()


@pytest.fixture
def pyfixest_model(df):
    """pyfixest Feols result."""
    return pyfixest.feols("y ~ x", data=df)


def test_adapt_statsmodels_passthrough(statsmodels_model):
    """Test that statsmodels models pass through without wrapping."""
    adapted = adapt_model(statsmodels_model)

    # Should be the same object, not wrapped
    assert adapted is statsmodels_model


def test_adapt_pyfixest_wraps(pyfixest_model):
    """Test that pyfixest models are wrapped with PyFixestAdapter."""
    adapted = adapt_model(pyfixest_model)

    assert isinstance(adapted, PyFixestAdapter)
    assert adapted._model is pyfixest_model


def test_pyfixest_adapter_params(pyfixest_model):
    """Test that PyFixestAdapter.params works correctly."""
    adapter = PyFixestAdapter(pyfixest_model)

    params = adapter.params
    assert isinstance(params, pd.Series)
    assert "x" in params.index
    assert "Intercept" in params.index


def test_pyfixest_adapter_bse(pyfixest_model):
    """Test that PyFixestAdapter.bse works correctly."""
    adapter = PyFixestAdapter(pyfixest_model)

    bse = adapter.bse
    assert isinstance(bse, pd.Series)
    assert "x" in bse.index
    assert "Intercept" in bse.index


def test_pyfixest_adapter_pvalues(pyfixest_model):
    """Test that PyFixestAdapter.pvalues works correctly."""
    adapter = PyFixestAdapter(pyfixest_model)

    pvalues = adapter.pvalues
    assert isinstance(pvalues, pd.Series)
    assert "x" in pvalues.index
    assert "Intercept" in pvalues.index


def test_pyfixest_adapter_rsquared(pyfixest_model):
    """Test that PyFixestAdapter.rsquared works correctly."""
    adapter = PyFixestAdapter(pyfixest_model)

    rsquared = adapter.rsquared
    assert isinstance(rsquared, float)
    assert 0 <= rsquared <= 1


def test_pyfixest_adapter_rsquared_adj(pyfixest_model):
    """Test that PyFixestAdapter.rsquared_adj works correctly."""
    adapter = PyFixestAdapter(pyfixest_model)

    rsquared_adj = adapter.rsquared_adj
    assert isinstance(rsquared_adj, float)
    assert 0 <= rsquared_adj <= 1


def test_pyfixest_adapter_nobs(pyfixest_model):
    """Test that PyFixestAdapter.nobs works correctly."""
    adapter = PyFixestAdapter(pyfixest_model)

    nobs = adapter.nobs
    assert isinstance(nobs, int)
    assert nobs == 6  # Our test data has 6 observations


def test_adapt_unsupported_type():
    """Test that adapt_model raises TypeError for unsupported models."""

    class FakeModel:
        pass

    fake_model = FakeModel()

    with pytest.raises(TypeError, match="Unsupported model type"):
        adapt_model(fake_model)


def test_interface_consistency(statsmodels_model, pyfixest_model):
    """Test that adapted models have consistent interfaces."""
    sm_adapted = adapt_model(statsmodels_model)
    pf_adapted = adapt_model(pyfixest_model)

    # Both should have the same attributes
    for attr in ["params", "bse", "pvalues", "rsquared", "rsquared_adj", "nobs"]:
        assert hasattr(sm_adapted, attr)
        assert hasattr(pf_adapted, attr)

    # Both should return the same types for Series attributes
    assert type(sm_adapted.params) == type(pf_adapted.params)
    assert type(sm_adapted.bse) == type(pf_adapted.bse)
    assert type(sm_adapted.pvalues) == type(pf_adapted.pvalues)
    assert type(sm_adapted.rsquared) == type(pf_adapted.rsquared)
    assert type(sm_adapted.rsquared_adj) == type(pf_adapted.rsquared_adj)
    # nobs can be int or float, both are numeric
    assert isinstance(sm_adapted.nobs, (int, float))
    assert isinstance(pf_adapted.nobs, (int, float))
    # But they should have the same value
    assert int(sm_adapted.nobs) == int(pf_adapted.nobs)
