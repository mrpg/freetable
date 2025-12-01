"""Tests for the tabularray() function."""

import pandas as pd
import pytest
import statsmodels.formula.api as smf

from freetable import tabularray


@pytest.fixture
def df():
    """Simple test dataframe."""
    return pd.DataFrame(
        {
            "y": [1, 2, 3, 4, 5, 6],
            "x1": [2, 4, 6, 8, 10, 12],
            "x2": [1, 1, 2, 2, 3, 3],
        }
    )


@pytest.fixture
def model1(df):
    """Single regression model."""
    return smf.ols("y ~ x1", data=df).fit()


@pytest.fixture
def model2(df):
    """Two-variable regression model."""
    return smf.ols("y ~ x1 + x2", data=df).fit()


def test_basic_tabularray(model1):
    """Test basic tabularray generation with single model."""
    result = tabularray(model1)

    # Check tabularray-specific structure
    assert r"\begin{table}[h!]" in result
    assert r"\begin{booktabs}[" in result
    assert r"\end{booktabs}" in result
    assert r"\end{table}" in result

    # Check tabularray templates
    assert r"\DeclareTblrTemplate{caption-sep}{default}" in result
    assert r"\DeclareTblrTemplate{remark-sep}{default}" in result
    assert r"\SetTblrStyle{remark}" in result
    assert r"\setlength{\customcolwidth}" in result

    # Check content
    assert "Intercept" in result
    assert "x1" in result
    assert r"$R^2$" in result
    assert r"Adj. $R^2$" in result
    assert "Observations" in result

    # Check tabularray preamble elements
    assert "caption =" in result
    assert "label =" in result
    assert "remark{}" in result
    assert "colspec =" in result
    assert "hline{1, Z}" in result

    # Check significance stars
    assert r"$^{***}$\,$p < 0.001$" in result
    assert r"$^{**}$\,$p < 0.01$" in result
    assert r"$^{*}$\,$p < 0.05$" in result


def test_multiple_models_tabularray(model1, model2):
    """Test tabularray with multiple models."""
    result = tabularray([model1, model2])

    # Check both models appear
    assert "Model 1" in result
    assert "Model 2" in result

    # Check x2 appears (only in model2)
    assert "x2" in result


def test_model_names_tabularray(model1, model2):
    """Test custom model names."""
    result = tabularray([model1, model2], model_names=["Treatment", "Control"])

    assert "Treatment" in result
    assert "Control" in result
    assert "Model 1" not in result


def test_digits_tabularray(model1):
    """Test digits parameter."""
    result_3 = tabularray(model1, digits=3)
    result_2 = tabularray(model1, digits=2)

    # Both should work
    assert r"\begin{booktabs}" in result_3
    assert r"\begin{booktabs}" in result_2


def test_caption_and_label_tabularray(model1):
    """Test custom caption and label."""
    result = tabularray(model1, caption="My Custom Caption", label="tab:mycustom")

    assert "caption = {My Custom Caption}" in result
    assert "label = {tab:mycustom}" in result


def test_rename_tabularray(model1, model2):
    """Test variable renaming."""
    result = tabularray(
        [model1, model2], rename={"x1": "Predictor One", "x2": "Predictor Two"}
    )

    assert "Predictor One" in result
    assert "Predictor Two" in result


def test_custom_stars_tabularray(model1):
    """Test custom significance thresholds."""
    result = tabularray(model1, stars=(0.1, 0.05, 0.01))

    # Check custom thresholds in note (sorted order)
    assert r"$^{***}$\,$p < 0.01$" in result
    assert r"$^{**}$\,$p < 0.05$" in result
    assert r"$^{*}$\,$p < 0.1$" in result


def test_stars_sorting_tabularray(model1):
    """Test that star thresholds are sorted correctly."""
    result1 = tabularray(model1, stars=(0.01, 0.05, 0.1))
    result2 = tabularray(model1, stars=(0.1, 0.05, 0.01))

    # Both should produce identical output
    assert result1 == result2


def test_extra_rows_tabularray(model1, model2):
    """Test extra custom rows."""
    result = tabularray(
        [model1, model2],
        extra_rows={"Outcome": ["Y", "Y"], "SE type": ["HC3", "Clustered"]},
    )

    assert "Outcome" in result
    assert "SE type" in result
    assert "HC3" in result
    assert "Clustered" in result


def test_extra_rows_wrong_length_tabularray(model1, model2):
    """Test that extra_rows validates length."""
    with pytest.raises(ValueError, match="has 3 values but 2 models"):
        tabularray([model1, model2], extra_rows={"Wrong": ["A", "B", "C"]})


def test_custom_header_tabularray(model1, model2):
    """Test custom grouped headers with span=1."""
    result = tabularray(
        [model1, model2], custom_header=[("Group A", 1), ("Group B", 1)]
    )

    # With span=1, no cell specification is created (just plain headers)
    # Check header content appears
    assert "Group A" in result
    assert "Group B" in result

    # Check for hline specifications
    assert "hline{2}" in result


def test_custom_header_multicolumn_tabularray(model1, model2):
    """Test custom header with span > 1."""
    result = tabularray([model1, model2], custom_header=[("Both Models", 2)])

    # Check for multicolumn specification
    assert "cell{1}{2} = {c = 2}{halign = c}" in result
    assert "Both Models" in result


def test_custom_header_wrong_span_tabularray(model1, model2):
    """Test that custom_header validates span."""
    with pytest.raises(ValueError, match="spans sum to 3 but 2 models"):
        tabularray([model1, model2], custom_header=[("Wrong", 3)])


def test_placement_tabularray(model1):
    """Test custom table placement."""
    result = tabularray(model1, placement="htbp")

    assert r"\begin{table}[htbp]" in result
    assert r"\begin{table}[h!]" not in result


def test_col_width_tabularray(model1):
    """Test custom column width."""
    result = tabularray(model1, col_width="0.2\\textwidth")

    assert r"0.2\textwidth" in result
    assert r"0.175\textwidth" not in result


def test_single_model_as_list_tabularray(model1):
    """Test that single model can be passed as list."""
    result1 = tabularray(model1)
    result2 = tabularray([model1])

    # Both should work
    assert r"\begin{booktabs}" in result1
    assert r"\begin{booktabs}" in result2


def test_intercept_renamed_tabularray(model1):
    """Test that Intercept is shown with default label."""
    result = tabularray(model1)

    assert "Intercept" in result


def test_cell_specs_for_stars_tabularray(model1):
    """Test that cells with stars get appto specifications."""
    result = tabularray(model1)

    # Should have cell specifications with appto for stars
    assert "appto = " in result


def test_hline_specs_tabularray(model1):
    """Test horizontal line specifications."""
    result = tabularray(model1)

    # Check for hline specifications
    assert "hline{1, Z}" in result
    assert r"\heavyrulewidth" in result
    assert r"\lightrulewidth" in result


def test_custom_header_hlines_tabularray(model1, model2):
    """Test hline specifications for custom headers."""
    result = tabularray(
        [model1, model2], custom_header=[("Group A", 1), ("Group B", 1)]
    )

    # Should have hline specs for the header rows
    assert "hline{2}" in result
    assert "leftpos = -1" in result


def test_custom_header_last_group_hline_tabularray(model1, model2):
    """Test that last group in custom header gets different hline formatting."""
    result = tabularray(
        [model1, model2], custom_header=[("Group A", 1), ("Group B", 1)]
    )

    # Last group should not have rightpos and endpos
    lines = result.split("\n")
    hline_specs = [
        line for line in lines if "hline{2}" in line and "lightrulewidth" in line
    ]

    # Should have at least one without endpos (the last one)
    has_without_endpos = any(
        "leftpos = -1}" in spec and "endpos" not in spec for spec in hline_specs
    )
    assert has_without_endpos


def test_row_guard_with_valign_tabularray(model1):
    """Test that header rows have guard and valign."""
    result = tabularray(model1)

    assert "row{1-1} = {guard, valign = m}" in result


def test_observations_centered_tabularray(model1):
    """Test that Observations row and below use centered columns."""
    result = tabularray(model1)

    # Should have cell specification to override S columns for lower rows
    assert "cell{" in result
    assert "} = {c}" in result


def test_combined_features_tabularray(model1, model2):
    """Test all features combined."""
    result = tabularray(
        [model1, model2],
        model_names=["Model A", "Model B"],
        digits=2,
        caption="Combined Test",
        label="tab:combined",
        rename={"x1": "Variable X"},
        stars=(0.1, 0.05, 0.01),
        extra_rows={"Type": ["OLS", "OLS"]},
        custom_header=[("Group", 2)],
        placement="htbp",
        col_width="0.2\\textwidth",
    )

    # Check all features are present
    assert "Model A" in result
    assert "Model B" in result
    assert "caption = {Combined Test}" in result
    assert "label = {tab:combined}" in result
    assert "Variable X" in result
    assert r"$^{***}$\,$p < 0.01$" in result
    assert "Type" in result
    assert "Group" in result
    assert r"\begin{table}[htbp]" in result
    assert r"0.2\textwidth" in result


def test_hborder_specifications_tabularray(model1):
    """Test that hborder specifications are present."""
    result = tabularray(model1)

    assert "hborder{1}" in result
    assert "hborder{Z}" in result
    assert r"\abovetopsep" in result
    assert r"\belowbottomsep" in result
    assert r"\belowrulesep" in result
    assert r"\aboverulesep" in result


def test_column_specifications_tabularray(model1):
    """Test column specifications."""
    result = tabularray(model1)

    assert "column{1} = {leftsep = 0pt}" in result
    assert "column{Z} = {rightsep = 0pt}" in result
