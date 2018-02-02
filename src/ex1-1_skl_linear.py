import numpy as np
import pandas as pd
import sklearn.linear_model
import plotly.offline as py
import plotly.graph_objs as go
from beeprint import pp

oecd_bli_df = pd.read_csv("../datasets/lifesat/oecd_bli_2015.csv", thousands=",")

gdp_per_capi_df = pd.read_csv("../datasets/lifesat/gdp_per_capita.csv",
                              thousands=",",
                              delimiter="\t",
                              encoding="latin1",
                              na_values="n/a")


# pp(oecd_bli_df.head())
# pp(gdp_per_capi_df.head())


def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"] == "TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]


country_stats_df = prepare_country_stats(oecd_bli_df, gdp_per_capi_df)
X = np.c_[country_stats_df["GDP per capita"]]
y = np.c_[country_stats_df["Life satisfaction"]]

lin_reg_model = sklearn.linear_model.LinearRegression()

lin_reg_model.fit(X, y)

X_new = np.array([[22587]])

y_hat = lin_reg_model.predict(X_new)

py.plot([
    go.Scatter(x=country_stats_df["GDP per capita"], y=country_stats_df["Life satisfaction"], name="data", mode='markers'),
    go.Scatter(x=X_new.flatten(), y=y_hat.flatten(), name="pred", mode='markers'),
], filename="/tmp/country_stats.html")
