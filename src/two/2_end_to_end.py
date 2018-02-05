import hashlib
import os, tarfile, pandas as pd, numpy as np
import plotly.figure_factory as ff
import plotly.offline as py

from beeprint import pp

import cufflinks as cf
from numpy.polynomial import Polynomial
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, LabelBinarizer, StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor

from two.transformers import CombinedAttributesAdder, CategoricalEncoder, IndexSelector, ColumnToIndexTransformer, Inspector, PassThrough


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


def get_num_df(df):
    return df.drop("ocean_proximity", axis=1)


def clean_df(df):
    def remove_nan_rows(df):
        df.dropna(subset=["total_bedrooms"], inplace=True)

        return df, np.nan

    def remove_feature(df):
        df.drop("total_bedrooms", axis=1, inplace=True)

        return df, np.nan

    def fill_nan(df):
        m = df["total_bedrooms"].median()
        df["total_bedrooms"].fillna(m, inplace=True)

        return df, m

    def impute_nan(df):
        num_df = get_num_df(df)

        imputer = Imputer(strategy="median")

        imputer.fit(num_df)

        X = imputer.transform(num_df)

        new_df = pd.DataFrame(X, columns=df.columns)

        return new_df, np.nan

    return remove_nan_rows(df)


def encode_df(df):
    def manual(df):
        l_encoder = LabelEncoder()
        housing_cat = df["ocean_proximity"]
        housing_cat_encoded = l_encoder.fit_transform(housing_cat)

        oh_encoder = OneHotEncoder()
        housing_cat_1hot = oh_encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))

        return housing_cat_1hot

    def auto(df):
        housing_cat = df["ocean_proximity"]
        encoder = LabelBinarizer(sparse_output=True)
        housing_cat_1hot = encoder.fit_transform(housing_cat)

        return housing_cat_1hot

    return auto(df)


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


def split(X, y):
    # train_set, test_set = split_train_test(df, test_ratio=0.2)

    # train_set, test_set = split_train_test_by_id(df, test_ratio=0.2, id_column="index")

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

    # train_sets, test_sets = stratified_train_test_split(df)
    #  train_set, test_set = train_sets[0], test_sets[0]

    return train_X, test_X, train_y, test_y


def get_X_y(df, x_cols, y_cols):
    X = df[x_cols].values
    y = df[y_cols].values

    return X, y


def get_X_pipeline(num_ix, cat_ix):
    num_pipeline = Pipeline([
        ('selector', IndexSelector(num_ix)),
        # ('inspector', Inspector('num_inspector')),
        ('imputer', Imputer(strategy='median')),
        ('attribs_adder', CombinedAttributesAdder()),
        ('poly', PolynomialFeatures(degree=1)),
        # ('std_scaler', MinMaxScaler())
    ])

    cat_pipeline = Pipeline([
        ('selector', IndexSelector(cat_ix)),
        # ('inspector', Inspector('cat_inspector')),
        ('label_binarizer', CategoricalEncoder(encoding="onehot-dense")),
    ])

    union_pipeline = FeatureUnion(transformer_list=[
        ('num_pipeline', num_pipeline),
        ('cat_pipeline', cat_pipeline),
    ])

    p = Pipeline([
        # ('inspector:before', Inspector('top_inspector:before')),
        ('union', union_pipeline),
        # ('inspector:after', Inspector('top_inspector:after')),
    ])

    return p


def get_y_pipeline():
    p = Pipeline([
        # ('std_scaler', MinMaxScaler()),
        ('pass_through', PassThrough())
    ])

    return p


def evaluate_error(model, y_pipeline, train_X, test_X, train_y, test_y):
    y_hat = model.predict(train_X).reshape(-1, 1)

    # train_y = y_pipeline.inverse_transform(train_y)
    # y_hat = y_pipeline.inverse_transform(y_hat)
    train_rmse = mean_squared_error(train_y, y_hat) ** 0.5

    print("train rmse: {}".format(train_rmse))

    y_hat = model.predict(test_X).reshape(-1, 1)

    # test_y = y_pipeline.inverse_transform(test_y)
    # y_hat = y_pipeline.inverse_transform(y_hat)

    test_rmse = mean_squared_error(test_y, y_hat) ** 0.5

    print("test rmse: {}".format(test_rmse))


def predict(model, y_pipeline, X, y_true):
    y_hat = model.predict(X).reshape(-1, 1)
    print("y_hat: \n")

    y_hat = y_pipeline.inverse_transform(y_hat)

    print(y_hat)
    print("y_true: \n")

    y_true = y_pipeline.inverse_transform(y_true)

    print(y_true)


def run():
    housing_df = load_housing_df()

    y_cols = ["median_house_value"]
    x_cols = [x for x in list(housing_df.columns) if x not in y_cols]
    cat_attribs = ["ocean_proximity"]
    num_attribs = [x for x in x_cols if x not in cat_attribs]

    X, y = get_X_y(housing_df, x_cols, y_cols)

    x_cti_trans = ColumnToIndexTransformer(full_columns=list(x_cols))

    cat_ix = x_cti_trans.transform(cat_attribs)
    num_ix = x_cti_trans.transform(num_attribs)

    train_X, test_X, train_y, test_y = split(X, y)

    x_pipeline = get_X_pipeline(num_ix, cat_ix)
    train_X = x_pipeline.fit_transform(train_X)
    test_X = x_pipeline.transform(test_X)

    y_pipeline = get_y_pipeline()
    train_y = y_pipeline.fit_transform(train_y)
    test_y = y_pipeline.transform(test_y)

    model = RandomForestRegressor(warm_start=False, bootstrap=False, max_features=6, n_estimators=80)

    def simple_evaluate(model, y_pipeline, train_X, test_X, train_y, test_y):
        model.fit(train_X, train_y)

        evaluate_error(model, y_pipeline, train_X, test_X, train_y, test_y)

        predict(model, y_pipeline, x_pipeline.transform(X[[17606, 18632, 14650, 3230, 3555]]), y_pipeline.transform(y[[17606, 18632, 14650, 3230, 3555]]))
        predict(model, y_pipeline, test_X[:5], test_y[:5])

    def cross_evaluate(model, y_pipeline, train_X, test_X, train_y, test_y):
        scores = cross_val_score(model, train_X, train_y, scoring="neg_mean_squared_error", cv=10)

        rmse_scores = np.sqrt(-scores)  # ** 0.5

        def display_scores(scores):
            print("Scores:", y_pipeline.inverse_transform(scores.reshape(-1, 1)))
            print("Mean:", y_pipeline.inverse_transform([[scores.mean()]]))
            print("Standard deviation:", y_pipeline.inverse_transform([[scores.std()]]))

        display_scores(rmse_scores)

    def grid_search(model, train_X, train_y):
        from sklearn.model_selection import GridSearchCV

        param_grid = [
            {'n_estimators': [3, 10, 30, 50, 80], 'max_features': [2, 4, 6, 8, 10]},
            {'bootstrap': [False], 'n_estimators': [3, 10, 30, 50, 80], 'max_features': [2, 4, 6, 8, 10]},
        ]

        grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=os.cpu_count(), scoring="neg_mean_squared_error")

        grid_search.fit(train_X, train_y)

        pp(grid_search.best_params_)

        pp(grid_search.best_estimator_)

        cvres = grid_search.cv_results_

        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            print(np.sqrt(-mean_score), params)

    def feature_importance(model, x_pipeline):
        i = model.feature_importances_

        extra_attrs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]

        # encoder = get_pipeline_object(x_pipeline, ["union", "cat_pipeline", "label_binarizer"])

        encoder = x_pipeline.get_params()["union__cat_pipeline__label_binarizer"]

        one_hot_classes = list(np.array(encoder.categories_).ravel())

        attrs = num_attribs + extra_attrs + one_hot_classes

        pp(sorted(zip(i, attrs), reverse=True))

    simple_evaluate(model, y_pipeline, train_X, test_X, train_y, test_y)

    feature_importance(model, x_pipeline)

    # cross_evaluate(model, y_pipeline, train_X, test_X, train_y, test_y)

    # grid_search(model, train_X, train_y)


run()
