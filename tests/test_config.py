"""Tests for the config module."""

import pandas as pd
import pytest
import statsmodels.formula.api as smf

import freetable
from freetable import config, table


@pytest.fixture
def reset_config():
    """Reset config to defaults after each test."""
    # Save original values
    original_values = {
        "model_prefix": config.model_prefix,
        "rsquared_label": config.rsquared_label,
        "adj_rsquared_label": config.adj_rsquared_label,
        "nobs_label": config.nobs_label,
        "intercept_label": config.intercept_label,
    }

    yield

    # Restore original values
    config.model_prefix = original_values["model_prefix"]
    config.rsquared_label = original_values["rsquared_label"]
    config.adj_rsquared_label = original_values["adj_rsquared_label"]
    config.nobs_label = original_values["nobs_label"]
    config.intercept_label = original_values["intercept_label"]


@pytest.fixture
def simple_model():
    """Create a simple regression model for testing."""
    df = pd.DataFrame(
        {
            "y": [1, 2, 3, 4, 5, 6],
            "x1": [2, 4, 6, 8, 10, 12],
        }
    )
    return smf.ols("y ~ x1", data=df).fit()


def test_config_default_values():
    """Test that config has correct default values."""
    assert config.model_prefix == "Model "
    assert config.rsquared_label == r"$R^2$"
    assert config.adj_rsquared_label == r"Adj. $R^2$"
    assert config.nobs_label == "Observations"
    assert config.intercept_label == "Intercept"


def test_config_accessible_from_freetable():
    """Test that config is accessible as freetable.config."""
    assert hasattr(freetable, "config")
    assert freetable.config is config


def test_config_modify_model_prefix(simple_model, reset_config):
    """Test modifying model_prefix config."""
    config.model_prefix = "M"
    result = table(simple_model)

    assert "M1" in result
    assert "Model 1" not in result


def test_config_modify_rsquared_label(simple_model, reset_config):
    """Test modifying rsquared_label config."""
    config.rsquared_label = "R-squared"
    result = table(simple_model)

    # Check that the custom label appears
    assert "R-squared &" in result
    # Check that it's not the default format (but Adj. $R^2$ will still contain $R^2$)
    lines = result.split("\n")
    rsquared_line = [line for line in lines if line.strip().startswith("R-squared")]
    assert len(rsquared_line) > 0


def test_config_modify_adj_rsquared_label(simple_model, reset_config):
    """Test modifying adj_rsquared_label config."""
    config.adj_rsquared_label = "Adjusted R-squared"
    result = table(simple_model)

    assert "Adjusted R-squared &" in result
    assert r"Adj. $R^2$ &" not in result


def test_config_modify_nobs_label(simple_model, reset_config):
    """Test modifying nobs_label config."""
    config.nobs_label = "N"
    result = table(simple_model)

    assert "N &" in result
    assert "Observations" not in result


def test_config_modify_intercept_label(simple_model, reset_config):
    """Test modifying intercept_label config."""
    config.intercept_label = "Constant"
    result = table(simple_model)

    assert "Constant &" in result
    # (Intercept) should not appear as a row label
    lines = [line.strip() for line in result.split("\n") if "&" in line]
    intercept_lines = [line for line in lines if line.startswith("(Intercept)")]
    assert len(intercept_lines) == 0


def test_config_modify_all_labels(simple_model, reset_config):
    """Test modifying all config labels at once."""
    config.model_prefix = "M"
    config.rsquared_label = "R2"
    config.adj_rsquared_label = "Adj. R2"
    config.nobs_label = "N"
    config.intercept_label = "Const"

    result = table(simple_model)

    assert "M1" in result
    assert "R2 &" in result
    assert "Adj. R2 &" in result
    assert "N &" in result
    assert "Const &" in result


def test_config_persists_across_multiple_calls(simple_model, reset_config):
    """Test that config changes persist across multiple table() calls."""
    config.model_prefix = "Model#"
    config.nobs_label = "Observations"

    result1 = table(simple_model)
    result2 = table(simple_model)

    assert "Model#1" in result1
    assert "Observations &" in result1
    assert "Model#1" in result2
    assert "Observations &" in result2


def test_config_multiple_models(simple_model, reset_config):
    """Test config with multiple models."""
    df = pd.DataFrame(
        {
            "y": [1, 2, 3, 4, 5, 6],
            "x1": [2, 4, 6, 8, 10, 12],
            "x2": [1, 1, 2, 2, 3, 3],
        }
    )
    model2 = smf.ols("y ~ x1 + x2", data=df).fit()

    config.model_prefix = "Specification "
    result = table([simple_model, model2])

    assert "Specification 1" in result
    assert "Specification 2" in result
    assert "Model 1" not in result
    assert "Model 2" not in result


def test_config_with_latex_formatting(simple_model, reset_config):
    """Test that config accepts LaTeX formatting."""
    config.rsquared_label = r"$R^{2}$"
    config.intercept_label = r"$\alpha$"

    result = table(simple_model)

    assert r"$R^{2}$ &" in result
    assert r"$\alpha$ &" in result


def test_config_empty_model_prefix(simple_model, reset_config):
    """Test config with empty model prefix."""
    config.model_prefix = ""
    result = table(simple_model)

    # Should just have numbers without prefix
    assert " & {1} \\\\" in result
    assert "Model " not in result


def test_config_class_instantiation():
    """Test that Config class can be instantiated with defaults."""
    from freetable.config import Config

    new_config = Config()
    assert new_config.model_prefix == "Model "
    assert new_config.rsquared_label == r"$R^2$"
    assert new_config.adj_rsquared_label == r"Adj. $R^2$"
    assert new_config.nobs_label == "Observations"
    assert new_config.intercept_label == "Intercept"
