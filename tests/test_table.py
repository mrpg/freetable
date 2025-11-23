"""Tests for the table() function."""

from pathlib import Path

import pandas as pd
import pytest
import statsmodels.formula.api as smf

from freetable import table


# Session-scoped fixture to collect all LaTeX outputs
@pytest.fixture(scope="session")
def latex_outputs():
    """Collect all LaTeX outputs for compilation test."""
    return []


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


def collect_latex(latex_outputs, test_name, latex_code):
    """Helper to collect LaTeX output with test name."""
    latex_outputs.append((test_name, latex_code))
    return latex_code


def test_basic_table(model1, latex_outputs):
    """Test basic table generation with single model."""
    result = collect_latex(latex_outputs, "Basic Table", table(model1))

    # Check structure
    assert r"\begin{table}[htbp]" in result
    assert r"\begin{threeparttable}" in result
    assert r"\begin{tabular}" in result
    assert r"\toprule" in result
    assert r"\midrule" in result
    assert r"\bottomrule" in result
    assert r"\end{table}" in result

    # Check content
    assert "(Intercept)" in result
    assert "x1" in result
    assert r"$R^2$" in result
    assert r"Adj. $R^2$" in result
    assert "Observations" in result

    # Check significance stars
    assert r"$^{***}p<0.001$" in result
    assert r"$^{**}p<0.01$" in result
    assert r"$^{*}p<0.05$" in result


def test_multiple_models(model1, model2, latex_outputs):
    """Test table with multiple models."""
    result = collect_latex(latex_outputs, "Multiple Models", table([model1, model2]))

    # Check both models appear
    assert "Model 1" in result
    assert "Model 2" in result

    # Check x2 appears (only in model2)
    assert "x2" in result


def test_model_names(model1, model2):
    """Test custom model names."""
    result = table([model1, model2], model_names=["Treatment", "Control"])

    assert "Treatment" in result
    assert "Control" in result
    assert "Model 1" not in result


def test_digits(model1):
    """Test digits parameter."""
    result_3 = table(model1, digits=3)
    result_2 = table(model1, digits=2)

    # Both should work but may have different precision
    assert r"\begin{table}" in result_3
    assert r"\begin{table}" in result_2


def test_caption_and_label(model1):
    """Test custom caption and label."""
    result = table(model1, caption="My Custom Caption", label="tab:mycustom")

    assert r"\caption{My Custom Caption}" in result
    assert r"\label{tab:mycustom}" in result


def test_rename(model1, model2, latex_outputs):
    """Test variable renaming."""
    result = collect_latex(
        latex_outputs,
        "Variable Renaming",
        table([model1, model2], rename={"x1": "Predictor One", "x2": "Predictor Two"}),
    )

    assert "Predictor One" in result
    assert "Predictor Two" in result
    assert "x1 &" not in result  # Should not appear as row label


def test_custom_stars(model1):
    """Test custom significance thresholds."""
    result = table(model1, stars=(0.1, 0.05, 0.01))

    # Check custom thresholds in note (sorted order)
    assert r"$^{***}p<0.01$" in result
    assert r"$^{**}p<0.05$" in result
    assert r"$^{*}p<0.1$" in result


def test_stars_sorting(model1):
    """Test that star thresholds are sorted correctly regardless of input order."""
    result1 = table(model1, stars=(0.01, 0.05, 0.1))
    result2 = table(model1, stars=(0.1, 0.05, 0.01))

    # Both should produce identical output
    assert result1 == result2


def test_extra_rows(model1, model2, latex_outputs):
    """Test extra custom rows."""
    result = collect_latex(
        latex_outputs,
        "Extra Rows",
        table(
            [model1, model2],
            extra_rows={"Outcome": ["Y", "Y"], "SE type": ["HC3", "Clustered"]},
        ),
    )

    assert "Outcome & {Y} & {Y}" in result
    assert "SE type & {HC3} & {Clustered}" in result


def test_extra_rows_wrong_length(model1, model2):
    """Test that extra_rows validates length."""
    with pytest.raises(ValueError, match="has 3 values but 2 models"):
        table([model1, model2], extra_rows={"Wrong": ["A", "B", "C"]})


def test_custom_header(model1, model2):
    """Test custom grouped headers."""
    result = table([model1, model2], custom_header=[("Group A", 1), ("Group B", 1)])

    # For span=1, no multicolumn is used (just braces)
    assert "{Group A}" in result
    assert "{Group B}" in result
    assert r"\cmidrule(lr){2-2}" in result
    assert r"\cmidrule(lr){3-3}" in result


def test_custom_header_multicolumn(model1, model2, latex_outputs):
    """Test custom header with span > 1."""
    result = collect_latex(
        latex_outputs,
        "Custom Header Multicolumn",
        table([model1, model2], custom_header=[("Both Models", 2)]),
    )

    assert r"\multicolumn{2}{c}{{Both Models}}" in result
    assert r"\cmidrule(lr){2-3}" in result


def test_custom_header_wrong_span(model1, model2):
    """Test that custom_header validates span."""
    with pytest.raises(ValueError, match="spans sum to 3 but 2 models"):
        table([model1, model2], custom_header=[("Wrong", 3)])


def test_placement(model1):
    """Test custom table placement."""
    result = table(model1, placement="h!")

    assert r"\begin{table}[h!]" in result
    assert r"\begin{table}[htbp]" not in result


def test_resize_false(model1):
    """Test resize=False (default)."""
    result = table(model1, resize=False)

    assert r"\resizebox" not in result
    assert r"\begin{tabular}" in result


def test_resize_true(model1, latex_outputs):
    """Test resize=True."""
    result = collect_latex(latex_outputs, "Resize Table", table(model1, resize=True))

    assert r"\resizebox{\textwidth}{!}{%" in result
    assert r"\end{tabular}}" in result


def test_no_stars_for_nonsignificant(df):
    """Test that non-significant coefficients have no stars."""
    # Create a model with a non-significant coefficient
    df_nosig = pd.DataFrame(
        {
            "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "x": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "noise": [0.1, -0.1, 0.2, -0.2, 0.15, -0.15, 0.25, -0.25, 0.3, -0.3],
        }
    )
    model = smf.ols("y ~ x + noise", data=df_nosig).fit()
    result = table(model)

    # Should not have empty superscripts like ^{}
    assert "^{}" not in result


def test_combined_features(model1, model2, latex_outputs):
    """Test all features combined."""
    result = collect_latex(
        latex_outputs,
        "All Features Combined",
        table(
            [model1, model2],
            model_names=["Model A", "Model B"],
            digits=2,
            caption="Combined Test",
            label="tab:combined",
            rename={"x1": "Variable X"},
            stars=(0.1, 0.05, 0.01),
            extra_rows={"Type": ["OLS", "OLS"]},
            custom_header=[("Group", 2)],
            placement="h!",
            resize=True,
        ),
    )

    # Check all features are present
    assert "Model A" in result
    assert "Model B" in result
    assert r"\caption{Combined Test}" in result
    assert r"\label{tab:combined}" in result
    assert "Variable X" in result
    assert r"$^{***}p<0.01$" in result
    assert "Type & {OLS} & {OLS}" in result
    assert r"\multicolumn{2}{c}{{Group}}" in result
    assert r"\begin{table}[h!]" in result
    assert r"\resizebox" in result


def test_single_model_as_list(model1):
    """Test that single model can be passed as list."""
    result1 = table(model1)
    result2 = table([model1])

    # Both should work and produce similar output (model names differ)
    assert r"\begin{table}" in result1
    assert r"\begin{table}" in result2


def test_intercept_renamed(model1):
    """Test that Intercept is renamed to (Intercept)."""
    result = table(model1)

    assert "(Intercept)" in result
    # Should not appear as "Intercept" in row label position
    lines = result.split("\n")
    for line in lines:
        if line.strip().startswith("(Intercept)"):
            assert True
            break
    else:
        pytest.fail("(Intercept) not found as row label")


def test_zzz_write_latex_compilation_file(latex_outputs):
    """Write all collected LaTeX outputs to a compilable .tex file.

    This test runs last (zzz prefix) to ensure all outputs are collected.
    """
    output_file = Path(__file__).parent / "test_output.tex"

    # Create complete LaTeX document
    preamble = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{siunitx}
\usepackage{threeparttable}
\usepackage{graphicx}
\usepackage[margin=1in]{geometry}

\title{freetable Test Output}
\author{Generated by pytest}
\date{\today}

\begin{document}
\maketitle

\section{Introduction}
This document contains all LaTeX tables generated during the freetable test suite.
Each table demonstrates a different feature or combination of features.

"""

    epilogue = r"""
\end{document}
"""

    # Assemble all tables with sections
    with open(output_file, "w") as f:
        f.write(preamble)

        for i, (test_name, latex_code) in enumerate(latex_outputs, 1):
            f.write(f"\n\\section{{Test {i}: {test_name}}}\n\n")
            f.write(latex_code)
            f.write("\n")

        f.write(epilogue)

    # Verify file was created and has content
    assert output_file.exists()
    assert output_file.stat().st_size > 1000  # Should have substantial content

    print(f"\nLaTeX compilation test file written to: {output_file}")
    print(f"  Total tables: {len(latex_outputs)}")
    print(f"  File size: {output_file.stat().st_size} bytes")
    print("\nTo compile: cd tests && pdflatex test_output.tex")
