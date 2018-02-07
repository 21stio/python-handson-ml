from beeprint import pp
from plotly import tools
from sklearn.base import clone, BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import fetch_mldata
import numpy as np
from sklearn.linear_model import SGDClassifier
import plotly.graph_objs as go
import plotly.offline as py
from sklearn.preprocessing import StandardScaler


mnist = fetch_mldata('MNIST original')

X, y = mnist["data"], mnist["target"]

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

shuffle_index = np.random.permutation(60000)

X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)


def simple_training(model):
    model.fit(X_train, y_train_5)

    p = model.predict(X_train[:20])


def own_cross_training(model):
    skfolds = StratifiedKFold(n_splits=3, random_state=42)

    for train_index, test_index in skfolds.split(X_train, y_train_5):
        clone_clf = clone(model)

        X_train_folds = X_train[train_index]
        y_train_folds = y_train_5[train_index]

        X_test_folds = X_train[test_index]
        y_test_folds = y_train_5[test_index]

        clone_clf.fit(X_train_folds, y_train_folds)
        y_hat = clone_clf.predict(X_test_folds)

        n_correct = sum(y_hat == y_test_folds)

        print(n_correct / len(y_hat))


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()


class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass

    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


def cross_training(model, X_train, y_train):
    s = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy")
    pp(s)


def cross_eval(model, X_train, y_train):
    y_hat = cross_val_predict(model, X_train, y_train, cv=3)

    eval(y_train, y_hat)

    plot_confusion_matrix_heatmap(y_train, y_hat)


def eval(y_true, y_hat):
    cm = confusion_matrix(y_true, y_hat)

    pp(cm)

    ps = precision_score(y_true, y_hat, average='micro')

    rs = recall_score(y_true, y_hat, average='micro')

    f1 = f1_score(y_true, y_hat, average='micro')

    print("Precision: {:.2f}, Recall: {:.2f} F1 Score: {:.2f}".format(ps, rs, f1))


def plot_precision_recall_vs_threshold(y_true, y_scores, labels):
    data = []

    for i, y_score in enumerate(y_scores):
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)

        data.append(go.Scatter(x=thresholds, y=precisions[:-1], name=labels[i] + ' precision', mode='line'))
        data.append(go.Scatter(x=thresholds, y=recalls[:-1], name=labels[i] + ' recall', mode='line'))

    layout = go.Layout(
        title='Plot Title',
        xaxis=dict(
            title='Recall',
        ),
        yaxis=dict(
            title='Precision',
        )
    )
    fig = go.Figure(data=data, layout=layout)

    py.plot(fig, filename='/tmp/precision_recall_vs_threshold.html')


def predict_with_threshold(y_scores, y_true, threshold):
    y_hat = y_scores > threshold

    eval(y_true, y_hat)


def plot_roc_curve(y_true, y_scores, labels):
    data = []

    for i, y_score in enumerate(y_scores):
        fpr, tpr, thresholds = roc_curve(y_true, y_score)

        data.append(go.Scatter(x=fpr, y=tpr, name=labels[i], mode='line'))

    layout = go.Layout(
        title='Plot Title',
        xaxis=dict(
            title='False Positive Rate',
        ),
        yaxis=dict(
            title='True Positive Rate',
        )
    )
    fig = go.Figure(data=data, layout=layout)

    py.plot(fig, filename='/tmp/roc_curve.html')


def plot_confusion_matrix_heatmap(y_true, y_hat):
    cm = confusion_matrix(y_true, y_hat)

    py.plot([
        go.Heatmap(z=cm[::-1], name='', colorscale='Viridis'),
    ], filename='/tmp/confusion_matrix_heatmap.html')

    cmn = cm * 1
    np.fill_diagonal(cmn, 0)

    row_sums = cm.sum(axis=1, keepdims=True)

    cmn = cmn / row_sums

    py.plot([
        go.Heatmap(z=cmn[::-1], name='', colorscale='Viridis'),
    ], filename='/tmp/confusion_matrix_heatmap_normal.html')

def train_score(model, X_train, y_train):
    s = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
    pp(s)


def get_scores(model, X_train, y_train, method):
    scores = cross_val_predict(model, X_train, y_train, cv=3, method=method)


def run1():
    sgd_clf = SGDClassifier(random_state=42)

    sgd_y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method='decision_function')

    forest_clf = RandomForestClassifier(random_state=42)

    y_scores_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")[:, 1]

    plot_roc_curve(y_train_5, [sgd_y_scores, y_scores_forest], ['sgd', 'random forest'])

    plot_precision_recall_vs_threshold(y_train_5, [sgd_y_scores, y_scores_forest], ['sgd', 'random forest'])


sgd_clf = SGDClassifier(random_state=42)
forest_clf = RandomForestClassifier(random_state=42)

std = StandardScaler()

# X_train = std.fit_transform(X_train.astype(np.float64))

# train_score(sgd_clf, X_train, y_train)
# train_score(forest_clf, X_train, y_train)

cross_eval(forest_clf, X_train, y_train)
