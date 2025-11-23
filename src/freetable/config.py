"""Configuration object for customizable labels and strings."""


class Config:
    """Configuration object for customizable labels and strings.

    Attributes:
        model_prefix: Prefix for default model names (e.g., "Model " -> "Model 1", "Model 2")
        rsquared_label: Label for R-squared statistic
        adj_rsquared_label: Label for adjusted R-squared statistic
        nobs_label: Label for number of observations
        intercept_label: Label for intercept term in regression output

    Example:
        >>> import freetable
        >>> freetable.config.rsquared_label = "R-squared"
        >>> freetable.config.model_prefix = "Model "
    """

    def __init__(self) -> None:
        self.model_prefix = "Model "
        self.rsquared_label = r"$R^2$"
        self.adj_rsquared_label = r"Adj. $R^2$"
        self.nobs_label = "Observations"
        self.intercept_label = "(Intercept)"


config = Config()
