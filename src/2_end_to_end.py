import hashlib
import os, tarfile, pandas as pd, numpy as np
import plotly.figure_factory as ff
import plotly.offline as py

from beeprint import pp
from six.moves import urllib

import cufflinks as cf
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


cf.set_config_file(offline=True, theme='pearl')

HOUSING_PATH = "../datasets/housing"


def load_housing_df(housing_path=HOUSING_PATH):
    p = os.path.join(housing_path, "housing.csv")

    df = pd.read_csv(p)

    return df


def prep_housing_df(df):
    def prep_index(df):
        df.reset_index(inplace=True)

        df["id"] = df["longitude"] * 1000 + df["latitude"]

        return df

    def prep_income_cat(df):
        df["income_cat"] = np.ceil(df["median_income"] / 1.5)
        df["income_cat"].where(df["income_cat"] < 5, 5.0, inplace=True)
        return df

    def prep_feature_combis(df):
        df["rooms_per_household"] = df["total_rooms"] / df["households"]
        df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
        df["bedrooms_per_household"] = df["total_bedrooms"] / df["households"]
        df["population_per_household"] = df["population"] / df["households"]

        return df

    df = prep_index(df)
    df = prep_income_cat(df)
    df = prep_feature_combis(df)

    return df


def clean_df(df):




def visualize(df):
    df.iplot(kind='histogram', bins=50, subplots=True, filename='/tmp/histogram-subplots.html', asPlot=True)
    # df.scatter_matrix(filename='/tmp/scatter-matrix.html')

    df.iplot(
        kind="scatter",
        x="longitude",
        y="latitude",
        filename='/tmp/loc.html',
        asPlot=True,
    )

    fig = ff.create_scatterplotmatrix(df[["housing_median_age", "total_rooms", "median_income", "median_house_value", ]], diag='histogram', width=1000, height=1000)
    py.plot(fig, filename='/tmp/scatterplotmatrix.html')


def inspect(df):
    print("\n\nHEAD")
    pp(df.head())

    print("\n\nINFO")
    pp(df.info())

    print("\n\nINCOME_CAT_DIST")
    pp(df["income_cat"].value_counts() / len(df))

    print("\n\nCORR median_house_value")
    corr_matrix = df.corr()
    pp(corr_matrix["median_house_value"].sort_values(ascending=False))


def inspect_train_test_sets(train, test):
    print(len(train), "train +", len(test), "test")


def split_train_test(df, test_ratio):
    shuffled_indices = np.random.permutation(len(df))
    test_set_size = int(len(df) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    return df.iloc[train_indices], df.iloc[test_indices]


def test_set_check(identifier, test_ration, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ration


def split_train_test_by_id(df, test_ratio, id_column, hash=hashlib.md5):
    ids = df[id_column]
    in_test_set = ids.apply(lambda _id: test_set_check(_id, test_ratio, hash))

    return df.loc[~in_test_set], df.loc[in_test_set]


def stratified_train_test_split(df):
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    train_sets = []
    test_sets = []
    for train_indices, test_indices in split.split(df, df["income_cat"]):
        train_sets.append(df.loc[train_indices])
        test_sets.append(df.loc[test_indices])

    return train_sets, test_sets


def split(df):
    # train_set, test_set = split_train_test(df, test_ratio=0.2)

    # train_set, test_set = split_train_test_by_id(df, test_ratio=0.2, id_column="index")

    # train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

    train_sets, test_sets = stratified_train_test_split(df)
    train_set, test_set = train_sets[0], test_sets[0]

    return train_set, test_set


housing_df = load_housing_df()

housing_df = prep_housing_df(housing_df)

train_set, test_set = split(housing_df)

#visualize(housing_df)

inspect(housing_df)
# inspect(train_set)