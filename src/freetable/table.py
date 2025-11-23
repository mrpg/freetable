"""Table generation for statsmodels regression results."""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from freetable.config import config


def table(
    models: Union[Any, List[Any]],
    model_names: Optional[List[str]] = None,
    digits: int = 3,
    caption: str = "Regression Results",
    label: str = "tab:regression",
    rename: Optional[Dict[str, str]] = None,
    stars: Optional[Tuple[float, ...]] = None,
    extra_rows: Optional[Dict[str, Sequence[str]]] = None,
    custom_header: Optional[List[Tuple[str, int]]] = None,
    placement: str = "htbp",
    resize: bool = False,
) -> str:
    """Convert statsmodels regression results to a LaTeX table.

    Creates a publication-ready LaTeX table with coefficients, standard errors,
    significance stars, and model statistics. Uses threeparttable, siunitx,
    and booktabs for professional formatting.

    Args:
        models: A single statsmodels result object or a list of results.
        model_names: Optional list of names for each model column.
                    Defaults to ["Model 1", "Model 2", ...].
        digits: Number of decimal places for numerical output. Default: 3.
        caption: Table caption text. Default: "Regression Results".
        label: LaTeX label for cross-referencing. Default: "tab:regression".
        rename: Optional dictionary to rename variables in output.
               Example: {"x1": "Gender", "x2": "Age"}. Default: None.
        stars: Optional tuple of significance thresholds for stars.
              Order doesn't matter - will be sorted automatically.
              Default: (0.001, 0.01, 0.05) giving ***, **, *.
              Example: (0.1, 0.05, 0.01) or (0.01, 0.05, 0.1) both work.
        extra_rows: Optional dictionary of additional rows to add after statistics.
                   Keys are row labels, values are lists of entries for each model.
                   Example: {"Outcome": ["Y1", "Y2"], "SE": ["HC3", "HC3"]}.
        custom_header: Optional list of (label, span) tuples for grouped headers.
                      Applied left-to-right across model columns.
                      Example: [("Treatment", 2), ("Control", 1)] spans 2+1=3 models.
        placement: LaTeX table placement specifier. Default: "htbp".
                  Example: "h!" for here-definitely.
        resize: If True, resize table to fit text width using graphicx package.
               Default: False.

    Returns:
        A string containing the complete LaTeX table code.

    Example:
        >>> import statsmodels.formula.api as smf
        >>> import pandas as pd
        >>> from freetable import table
        >>>
        >>> df = pd.DataFrame({'y': [1, 2, 3, 4], 'x': [2, 4, 6, 8]})
        >>> m1 = smf.ols('y ~ x', data=df).fit()
        >>> m2 = smf.ols('y ~ x + z', data=df).fit()
        >>> latex = table(
        ...     [m1, m2],
        ...     caption="My Results",
        ...     rename={"x": "Predictor"},
        ...     extra_rows={"Outcome": ["Y", "Y"], "SE type": ["HC3", "HC3"]},
        ...     custom_header=[("Treatment", 1), ("Control", 1)],
        ...     placement="h!",
        ...     resize=True
        ... )
        >>> print(latex)

    Note:
        Default significance levels: *** p<0.001; ** p<0.01; * p<0.05
        Standard errors are shown in parentheses below coefficients.
        The Intercept parameter is renamed to "(Intercept)" in the output.

        Configurable labels: You can customize labels globally using the config object:
            >>> import freetable
            >>> freetable.config.rsquared_label = "R-squared"
            >>> freetable.config.nobs_label = "N"
            >>> freetable.config.model_prefix = "Model "
            >>> freetable.config.intercept_label = "Constant"
            >>> freetable.config.adj_rsquared_label = r"Adj. R$^2$"

        LaTeX special characters: This function does NOT escape special LaTeX
        characters in user inputs (caption, label, rename values, etc.). This
        is intentional - users can include LaTeX commands if desired. If your
        data contains special characters like $, %, &, _, you must escape them
        manually or they will be interpreted as LaTeX commands.
    """
    if not isinstance(models, list):
        models = [models]

    if model_names is None:
        model_names = [f"{config.model_prefix}{i+1}" for i in range(len(models))]

    if rename is None:
        rename = {}

    # Set default significance thresholds and sort them
    if stars is None:
        stars = (0.001, 0.01, 0.05)
    sorted_thresholds = sorted(stars)

    def get_stars(p: float) -> str:
        """Return significance stars based on p-value."""
        for i, threshold in enumerate(sorted_thresholds):
            if p < threshold:
                return "*" * (len(sorted_thresholds) - i)
        return ""

    # Collect all unique parameters across models
    all_params = []
    for m in models:
        all_params.extend(m.params.index)
    # Sort with Intercept first, then alphabetically
    all_params = sorted(set(all_params), key=lambda x: (x != "Intercept", x))

    # Build header rows
    if custom_header:
        # Validate custom header spans sum to number of models
        total_span = sum(span for _, span in custom_header)
        if total_span != len(models):
            raise ValueError(
                f"custom_header spans sum to {total_span} but {len(models)} models provided"
            )

        # Build multicolumn header row
        header_parts = []
        for group_label, span in custom_header:
            if span == 1:
                header_parts.append(f"{{{group_label}}}")
            else:
                header_parts.append(
                    f"\\multicolumn{{{span}}}{{c}}{{{{{group_label}}}}}"
                )
        multicolumn_header = " & ".join(header_parts)

        # Build cmidrule commands
        cmidrules = []
        col_start = 2  # Start at column 2 (column 1 is for variable names)
        for _, span in custom_header:
            col_end = col_start + span - 1
            cmidrules.append(f"\\cmidrule(lr){{{col_start}-{col_end}}}")
            col_start = col_end + 1
        cmidrule_line = " ".join(cmidrules)

        # Build regular model names row
        model_names_row = " & ".join(["{" + name + "}" for name in model_names])

        # Combine into header structure (don't add leading & since it's added in LaTeX assembly)
        header = f"{multicolumn_header} \\\\\n{cmidrule_line}\n & {model_names_row}"
    else:
        # Simple header without grouping
        header = " & ".join(["{" + name + "}" for name in model_names])

    # Build coefficient and standard error rows
    rows = []
    for param in all_params:
        coef_row = []
        se_row = []

        # Apply renaming: first handle Intercept, then check rename dict
        if param == "Intercept":
            param_name = config.intercept_label
        else:
            param_name = rename.get(param, param)

        for m in models:
            if param in m.params:
                coef = m.params[param]
                se = m.bse[param]
                p = m.pvalues[param]
                star_str = get_stars(p)
                # Only add superscript if there are stars
                if star_str:
                    coef_row.append(f"{coef:.{digits}f}^{{{star_str}}}")
                else:
                    coef_row.append(f"{coef:.{digits}f}")
                se_row.append(f"({se:.{digits}f})")
            else:
                coef_row.append("")
                se_row.append("")

        rows.append(f"{param_name} & " + " & ".join(coef_row) + r" \\")
        rows.append(" & " + " & ".join(se_row) + r" \\")

    # Build statistics rows
    stats_rows = []
    stats_rows.append(
        f"{config.rsquared_label} & "
        + " & ".join([f"{m.rsquared:.{digits}f}" for m in models])
        + r" \\"
    )
    stats_rows.append(
        f"{config.adj_rsquared_label} & "
        + " & ".join([f"{m.rsquared_adj:.{digits}f}" for m in models])
        + r" \\"
    )
    stats_rows.append(
        f"{config.nobs_label} & "
        + " & ".join([f"{{{int(m.nobs)}}}" for m in models])
        + r" \\"
    )

    # Add extra custom rows if provided
    if extra_rows:
        for row_label, row_values in extra_rows.items():
            if len(row_values) != len(models):
                raise ValueError(
                    f"extra_rows['{row_label}'] has {len(row_values)} values "
                    f"but {len(models)} models provided"
                )
            stats_rows.append(
                f"{row_label} & "
                + " & ".join([f"{{{v}}}" for v in row_values])
                + r" \\"
            )

    # Define column format for siunitx
    col_format = "l" + " S[table-format=3.6]" * len(models)

    # Build significance note based on sorted thresholds
    star_notes = []
    for i, threshold in enumerate(sorted_thresholds):
        star_str = "*" * (len(sorted_thresholds) - i)
        star_notes.append(f"$^{{{star_str}}}p<{threshold}$")
    significance_note = "; ".join(star_notes)

    # Assemble the complete LaTeX table
    rows_str = "\n".join(rows)
    stats_str = "\n".join(stats_rows)

    # Build tabular content
    if resize:
        tabular_open = rf"\resizebox{{\textwidth}}{{!}}{{%{chr(10)}\begin{{tabular}}{{{col_format}}}"
        tabular_close = r"\end{tabular}}"
    else:
        tabular_open = rf"\begin{{tabular}}{{{col_format}}}"
        tabular_close = r"\end{tabular}"

    latex = rf"""\begin{{table}}[{placement}]
\centering
\sisetup{{parse-numbers=false, table-text-alignment=center}}
\begin{{threeparttable}}
{tabular_open}
\toprule
 & {header} \\
\midrule
{rows_str}
\midrule
{stats_str}
\bottomrule
{tabular_close}
\begin{{tablenotes}}[flushleft]
\scriptsize{{\item {significance_note}}}
\end{{tablenotes}}
\end{{threeparttable}}
\caption{{{caption}}}
\label{{{label}}}
\end{{table}}"""

    return latex
