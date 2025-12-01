"""Example demonstrating pyfixest support in freetable."""

import pandas as pd
import pyfixest as pf

from freetable import tabularx

# Create sample data
df = pd.DataFrame(
    {
        "wage": [10, 12, 15, 20, 18, 22, 25, 28, 30, 35],
        "education": [12, 14, 16, 18, 16, 18, 20, 22, 20, 22],
        "experience": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "region": [1, 1, 2, 2, 3, 3, 1, 1, 2, 2],
    }
)

# Fit models with pyfixest
m1 = pf.feols("wage ~ education", data=df)
m2 = pf.feols("wage ~ education + experience", data=df)
m3 = pf.feols("wage ~ education + experience | region", data=df)

# Generate LaTeX table
latex = tabularx(
    [m1, m2, m3],
    model_names=["Model 1", "Model 2", "Fixed Effects"],
    caption="Wage Regression Results (pyfixest)",
    label="tab:pyfixest_example",
    digits=3,
    rename={"education": "Education (years)", "experience": "Experience (years)"},
    extra_rows={"Fixed Effects": ["No", "No", "Region"]},
    custom_header=[("OLS", 2), ("FE", 1)],
    resize=True,
)

print(latex)
