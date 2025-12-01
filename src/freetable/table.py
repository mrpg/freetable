"""Table generation for statsmodels and pyfixest regression results."""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from freetable.adapter import adapt_model
from freetable.config import config


class TableCell:
    """Represents a single cell in the table matrix."""

    def __init__(self, content: str, has_stars: bool = False, star_str: str = ""):
        self.content = content
        self.has_stars = has_stars
        self.star_str = star_str


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
    """Convert regression results (statsmodels or pyfixest) to a LaTeX table.

    Creates a publication-ready LaTeX table with coefficients, standard errors,
    significance stars, and model statistics. Uses threeparttable, siunitx,
    booktabs, and tabularx for professional formatting.

    Supports regression results from statsmodels and pyfixest.

    Args:
        models: A single regression result object or a list of results.
               Supports statsmodels and pyfixest (Feols) result objects.
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

    # Adapt models to unified interface
    models = [adapt_model(m) for m in models]

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
        for i, (group_label, span) in enumerate(custom_header):
            if span == 1:
                header_parts.append(f"{{{group_label}}}")
            else:
                # Use "c @{}" for all multicolumns
                header_parts.append(
                    f"\\multicolumn{{{span}}}{{c @{{}}}}{{{{{group_label}}}}}"
                )
        multicolumn_header = " & ".join(header_parts)

        # Build cmidrule commands
        cmidrules = []
        col_start = 2  # Start at column 2 (column 1 is for variable names)
        for i, (_, span) in enumerate(custom_header):
            col_end = col_start + span - 1
            # Use (l) for last cmidrule, (lr) for others
            trim = "(l)" if i == len(custom_header) - 1 else "(lr)"
            cmidrules.append(f"\\cmidrule{trim}{{{col_start}-{col_end}}}")
            col_start = col_end + 1
        cmidrule_line = " ".join(cmidrules)

        # Build regular model names row
        model_names_row = " & ".join(["{" + name + "}" for name in model_names])

        # Combine into header structure (don't add leading & since it's added in LaTeX assembly)
        header = (
            f"{multicolumn_header} \\\\  % !\n{cmidrule_line}\n & {model_names_row}"
        )
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

    # Define column format for tabularx with siunitx
    # X for first column, then S columns with custom width for each model
    num_models = len(models)
    s_cols = f"*{{{num_models}}}{{S[table-format = 3.5, table-column-width = 0.175\\textwidth]}}"
    col_format = f"@{{}} X {s_cols} @{{}}"

    # Build significance note based on sorted thresholds
    star_notes = []
    for i, threshold in enumerate(sorted_thresholds):
        star_str = "*" * (len(sorted_thresholds) - i)
        star_notes.append(f"$^{{{star_str}}}p<{threshold}$")
    significance_note = "; ".join(star_notes)

    # Assemble the complete LaTeX table
    rows_str = "\n".join(rows)
    stats_str = "\n".join(stats_rows)

    # Build tabularx content
    if resize:
        tabularx_open = rf"\resizebox{{\textwidth}}{{!}}{{%{chr(10)}\begin{{tabularx}}{{\textwidth}}{{{col_format}}}  % !"
        tabularx_close = r"\end{tabularx}}"
    else:
        tabularx_open = rf"\begin{{tabularx}}{{\textwidth}}{{{col_format}}}  % !"
        tabularx_close = r"\end{tabularx}"

    latex = rf"""\begin{{table}}[{placement}]
\caption{{{caption}}}
\label{{{label}}}
\sisetup{{parse-numbers=false}}
\begin{{threeparttable}}
{tabularx_open}
\toprule
 & {header} \\
\midrule
{rows_str}
\midrule
{stats_str}
\bottomrule
{tabularx_close}
\begin{{tablenotes}}[flushleft]
{{\item {significance_note}}}
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}"""

    return latex


def tabularray(
    models: Union[Any, List[Any]],
    model_names: Optional[List[str]] = None,
    digits: int = 3,
    caption: str = "Regression Results",
    label: str = "tab:regression",
    rename: Optional[Dict[str, str]] = None,
    stars: Optional[Tuple[float, ...]] = None,
    extra_rows: Optional[Dict[str, Sequence[str]]] = None,
    custom_header: Optional[List[Tuple[str, int]]] = None,
    placement: str = "h!",
    col_width: str = "0.175\\textwidth",
) -> str:
    """Convert regression results to a LaTeX table using tabularray package.

    Creates a publication-ready LaTeX table with coefficients, standard errors,
    significance stars, and model statistics using the tabularray package with
    booktabs and siunitx libraries.

    Unlike the standard table() function which uses threeparttable and tabularx,
    this function uses tabularray's advanced features for more precise control
    over cell styling and spacing. The table content is built as a matrix first,
    allowing cell-specific styling in the tabularray preamble.

    Supports regression results from statsmodels and pyfixest.

    Args:
        models: A single regression result object or a list of results.
               Supports statsmodels and pyfixest (Feols) result objects.
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
        placement: LaTeX table placement specifier. Default: "h!".
        col_width: Width for S columns in tabularray. Default: "0.175\\textwidth".

    Returns:
        A string containing the complete LaTeX table code using tabularray.

    Example:
        >>> import statsmodels.formula.api as smf
        >>> import pandas as pd
        >>> from freetable import tabularray
        >>>
        >>> df = pd.DataFrame({'y': [1, 2, 3, 4], 'x': [2, 4, 6, 8]})
        >>> m1 = smf.ols('y ~ x', data=df).fit()
        >>> m2 = smf.ols('y ~ x + z', data=df).fit()
        >>> latex = tabularray(
        ...     [m1, m2],
        ...     caption="My Results",
        ...     rename={"x": "Predictor"},
        ...     extra_rows={"Outcome": ["Y", "Y"], "SE type": ["HC3", "HC3"]},
        ...     custom_header=[("Treatment", 1), ("Control", 1)]
        ... )
        >>> print(latex)

    Note:
        Default significance levels: *** p<0.001; ** p<0.01; * p<0.05
        Standard errors are shown in parentheses below coefficients.
        The Intercept parameter is renamed to "(Intercept)" in the output.

        Requires: \\usepackage{tabularray}
                 \\UseTblrLibrary{booktabs}
                 \\UseTblrLibrary{siunitx}
    """
    if not isinstance(models, list):
        models = [models]

    # Adapt models to unified interface
    models = [adapt_model(m) for m in models]

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

    num_models = len(models)

    # Build the complete table matrix
    # Matrix structure: List[List[TableCell]] where matrix[row][col]
    # Column 0 is the row label, columns 1..n are model values
    matrix: List[List[TableCell]] = []

    # Build header rows
    if custom_header:
        # Validate custom header spans sum to number of models
        total_span = sum(span for _, span in custom_header)
        if total_span != len(models):
            raise ValueError(
                f"custom_header spans sum to {total_span} but {len(models)} models provided"
            )

        # First header row with group labels (multicolumn headers)
        header_row1 = [TableCell("")]  # Empty cell for row labels column
        for group_label, span in custom_header:
            # Repeat the content for each cell in the span (tabularray merges them)
            for _ in range(span):
                header_row1.append(TableCell(group_label))

        # Second header row with model names
        header_row2 = [TableCell("")]
        for name in model_names:
            header_row2.append(TableCell(name))

        matrix.append(header_row1)
        matrix.append(header_row2)
        header_rows = 2
    else:
        # Simple header without grouping
        header_row = [TableCell("")]
        for name in model_names:
            header_row.append(TableCell(name))
        matrix.append(header_row)
        header_rows = 1

    # Build coefficient and standard error rows
    for param in all_params:
        coef_row = []
        se_row = []

        # Apply renaming: first handle Intercept, then check rename dict
        if param == "Intercept":
            param_name = config.intercept_label
        else:
            param_name = rename.get(param, param)

        coef_row.append(TableCell(param_name))
        se_row.append(TableCell(""))

        for m in models:
            if param in m.params:
                coef = m.params[param]
                se = m.bse[param]
                p = m.pvalues[param]
                star_str = get_stars(p)
                # Store coefficient with star info
                coef_row.append(
                    TableCell(
                        f"{coef:.{digits}f}",
                        has_stars=bool(star_str),
                        star_str=star_str,
                    )
                )
                se_row.append(TableCell(f"({se:.{digits}f})"))
            else:
                coef_row.append(TableCell(""))
                se_row.append(TableCell(""))

        matrix.append(coef_row)
        matrix.append(se_row)

    # Build statistics rows
    stats_row_start = len(matrix)

    rsq_row = [TableCell(config.rsquared_label)]
    for m in models:
        rsq_row.append(TableCell(f"{m.rsquared:.{digits}f}"))
    matrix.append(rsq_row)

    adj_rsq_row = [TableCell(config.adj_rsquared_label)]
    for m in models:
        adj_rsq_row.append(TableCell(f"{m.rsquared_adj:.{digits}f}"))
    matrix.append(adj_rsq_row)

    # Observations row and below should use centered columns, not S columns
    nobs_row_start = len(matrix)

    nobs_row = [TableCell(config.nobs_label)]
    for m in models:
        nobs_row.append(TableCell(f"{int(m.nobs)}"))
    matrix.append(nobs_row)

    # Add extra custom rows if provided
    if extra_rows:
        for row_label, row_values in extra_rows.items():
            if len(row_values) != len(models):
                raise ValueError(
                    f"extra_rows['{row_label}'] has {len(row_values)} values "
                    f"but {len(models)} models provided"
                )
            extra_row = [TableCell(row_label)]
            for v in row_values:
                extra_row.append(TableCell(f"{v}"))
            matrix.append(extra_row)

    # Now build tabularray cell specifications by scanning the matrix
    cell_specs = []
    for row_idx, row in enumerate(matrix):
        for col_idx, cell in enumerate(row):
            if cell.has_stars and col_idx > 0:  # Skip row label column
                # Convert to 1-indexed for tabularray
                tbl_row = row_idx + 1
                tbl_col = col_idx + 1
                cell_specs.append(
                    f"cell{{{tbl_row}}}{{{tbl_col}}} = {{appto = {{^{{{cell.star_str}}}}}}}"
                )

    # Handle custom header multicolumn specifications
    if custom_header:
        col_start = 2  # Start at column 2 (column 1 is for variable names)
        for group_label, span in custom_header:
            col_end = col_start + span - 1
            if span > 1:
                cell_specs.insert(
                    0,
                    f"cell{{1}}{{{col_start}}} = {{c = {span}}}{{halign = c}}",
                )
            col_start = col_end + 1

    # Build colspec
    colspec = f"X *{{{num_models}}}{{S[table-format = 3.5, table-column-width = {col_width}]}}"

    # Build hline specifications
    len(matrix)
    midrule_row1 = header_rows + 1  # After headers
    midrule_row2 = stats_row_start + 1  # Before statistics

    # Build tabularray preamble - split into table options and format specs
    # Table-level options go in [...]
    table_options = []
    table_options.append(f"caption = {{{caption}}},")
    table_options.append("footsep = \\belowbottomsep,")
    table_options.append("headsep = \\abovetopsep,")
    table_options.append(f"label = {{{label}}},")
    table_options.append("long,")
    table_options.append("postsep = 0pt,")
    table_options.append("presep = 0pt,")

    # Build significance note
    star_notes = []
    for i, threshold in enumerate(sorted_thresholds):
        star_str = "*" * (len(sorted_thresholds) - i)
        star_notes.append(f"$^{{{star_str}}}$\\,$p < {threshold}$")
    significance_note = ";\\: ".join(star_notes)
    table_options.append(f"remark{{}} = {{{significance_note}.}},")

    # Format specifications go in {...}
    format_specs = []

    # Add cell specifications
    format_specs.extend([spec + "," for spec in cell_specs])

    # Add colspec and other specifications
    format_specs.append(f"colspec = {{{colspec}}},")
    format_specs.append("column{1} = {leftsep = 0pt},")
    format_specs.append("column{Z} = {rightsep = 0pt},")
    format_specs.append(
        "hborder{1} = {abovespace = \\abovetopsep, belowspace = \\belowrulesep},"
    )
    format_specs.append(
        "hborder{Z} = {abovespace = \\aboverulesep, belowspace = \\belowbottomsep},"
    )
    format_specs.append(
        f"hborder{{{midrule_row1}, {midrule_row2}}} = {{abovespace = \\aboverulesep, belowspace = \\belowrulesep}},"
    )
    format_specs.append("hline{1, Z} = \\heavyrulewidth,")

    # Add hline specifications for custom headers
    if custom_header:
        hline_specs = []
        col_start = 2
        for i, (_, span) in enumerate(custom_header):
            col_end = col_start + span - 1
            # Format column range: single column is {4}, multiple is {2-3}
            col_range = f"{col_start}" if span == 1 else f"{col_start}-{col_end}"
            # Last group gets different formatting
            is_last = i == len(custom_header) - 1
            if is_last:
                hline_specs.append(
                    f"hline{{2}} = {{{col_range}}}{{\\lightrulewidth, leftpos = -1}}"
                )
            else:
                hline_specs.append(
                    f"hline{{2}} = {{{col_range}}}{{\\lightrulewidth, leftpos = -1, rightpos = -1, endpos}}"
                )
            col_start = col_end + 1
        format_specs.extend([spec + "," for spec in hline_specs])

    format_specs.append(f"hline{{{midrule_row1}, {midrule_row2}}} = \\lightrulewidth,")
    format_specs.append(f"row{{1-{header_rows}}} = {{guard, valign = m}},")

    # Override S column type for Observations and below - use centered columns instead
    nobs_row_tbl = nobs_row_start + 1  # Convert to 1-indexed
    last_row_tbl = len(matrix)
    format_specs.append(
        f"cell{{{nobs_row_tbl}-{last_row_tbl}}}{{2-{num_models + 1}}} = {{c}},"
    )

    format_specs.append("width = \\textwidth,")

    # Build the table body
    body_lines = []
    for row in matrix:
        row_content = []
        for cell in row:
            if cell.content:
                # Wrap non-empty cells in braces if they're in data columns
                row_content.append(f"{{{cell.content}}}" if cell.content else "")
            else:
                row_content.append("")
        body_lines.append("\t\t" + " & ".join(row_content) + " \\\\")

    # Assemble the complete LaTeX table
    table_options_str = "\n\t\t".join(table_options)
    format_specs_str = "\n\t\t".join(format_specs)
    body_str = "\n".join(body_lines)

    latex = f"""\\begin{{table}}[{placement}]
\t\\sisetup{{parse-numbers=false}}
\t\\DeclareTblrTemplate{{caption-sep}}{{default}}{{: \\strut}}
\t\\DeclareTblrTemplate{{remark-sep}}{{default}}{{\\strut}}
\t\\SetTblrStyle{{remark}}{{font = \\footnotesize, halign = j, hang = 0pt, indent = 0pt}}
\t\\setlength{{\\customcolwidth}}{{{col_width}}}
\t\\begin{{booktabs}}[
\t\t{table_options_str}
\t]{{
\t\t{format_specs_str}
\t}}
{body_str}
\t\\end{{booktabs}}
\\end{{table}}"""

    return latex
