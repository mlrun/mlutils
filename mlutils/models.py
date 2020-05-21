from inspect import signature, _empty
from importlib import import_module

import json

from mlrun.artifacts import PlotArtifact
from .plots import (learning_curves,
                    feature_importances,
                    precision_recall_bin,
                    precision_recall_multi,
                    roc_multi,
                    roc_bin,
                    confusion_matrix)

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
from scikitplot.metrics import plot_calibration_curve
import matplotlib.pyplot as plt


def get_class_fit(module_pkg_class: str):
    """generate a model config
    :param module_pkg_class:  str description of model, e.g.
        `sklearn.ensemble.RandomForestClassifier`
    """
    splits = module_pkg_class.split(".")
    model_ = getattr(import_module(".".join(splits[:-1])), splits[-1])
    f = list(signature(model_().fit).parameters.items())
    d = {}
    for i in range(len(f)):
        d.update({f[i][0]: None if f[i][1].default is _empty
                  else f[i][1].default})

    return ({"CLASS": model_().get_params(),
             "FIT": d,
             "META": {"pkg_version": import_module(splits[0]).__version__,
                      "class": module_pkg_class}})


def create_class(pkg_class: str):
    """Create a class from a package.module.class string

    :param pkg_class:  full class location,
                       e.g. "sklearn.model_selection.GroupKFold"
    """
    splits = pkg_class.split(".")
    clfclass = splits[-1]
    pkg_module = splits[:-1]
    class_ = getattr(import_module(".".join(pkg_module)), clfclass)
    return class_


def create_function(pkg_func: list):
    """Create a function from a package.module.function string

    :param pkg_func:  full function location,
                      e.g. "sklearn.feature_selection.f_classif"
    """
    splits = pkg_func.split(".")
    pkg_module = ".".join(splits[:-1])
    cb_fname = splits[-1]
    pkg_module = __import__(pkg_module, fromlist=[cb_fname])
    function_ = getattr(pkg_module, cb_fname)
    return function_


def gen_sklearn_model(model_pkg, skparams):
    """generate an sklearn model configuration

    input can be either a "package.module.class" or
    a json file
    """
    if model_pkg.endswith("json"):
        model_config = json.load(open(model_pkg, "r"))
    else:
        model_config = get_class_fit(model_pkg)

    for k, v in skparams:
        if k.startswith('CLASS_'):
            model_config['CLASS'][k[6:]] = v
        if k.startswith('FIT_'):
            model_config['FIT'][k[4:]] = v

    return model_config


def eval_class_model(
    context,
    xtest,
    ytest,
    model,
    plots_dest: str = "plots",
    pred_params: dict = {}
):
    """generate predictions and validation stats

    pred_params are non-default, scikit-learn api prediction-function
    parameters. For example, a tree-type of model may have a tree depth
    limit for its prediction function.

    :param xtest:        features array type Union(DataItem, DataFrame,
                         numpy array)
    :param ytest:        ground-truth labels Union(DataItem, DataFrame,
                         Series, numpy array, List)
    :param model:        estimated model
    :param pred_params:  (None) dict of predict function parameters
    """
    if isinstance(ytest, np.ndarray):
        unique_labels = np.unique(ytest)
    elif isinstance(ytest, list):
        unique_labels = set(ytest)
    else:
        try:
            ytest = ytest.values
            unique_labels = np.unique(ytest)
        except Exception as e:
            raise Exception(f"unrecognized data type for ytest {e}")

    n_classes = len(unique_labels)
    is_multiclass = True if n_classes > 2 else False

    # INIT DICT...OR SOME OTHER COLLECTOR THAT CAN BE ACCESSED
    mm_plots = []
    mm_tables = []
    mm = {}

    ypred = model.predict(xtest, **pred_params)
    mm.update({
        "test-accuracy": float(metrics.accuracy_score(ytest, ypred)),
        "test-error": np.sum(ytest != ypred) / ytest.shape[0]})

    # GEN PROBS (INCL CALIBRATED PROBABILITIES)
    if hasattr(model, "predict_proba"):
        yprob = model.predict_proba(xtest, **pred_params)
        if not is_multiclass:
            plot_calibration_curve(ytest, [yprob], ['xgboost'])
            context.log_artifact(PlotArtifact("calibration curve", body=plt.gcf()),
                             local_path=f"{plots_dest}/calibration curve.html")
    else:
        # todo if decision fn...
        raise Exception("not implemented for this classifier")

    # start evaluating:
    # mm_plots.extend(learning_curves(model))
    if hasattr(model, "evals_result"):
        results = model.evals_result()
        train_set = list(results.items())[0]
        valid_set = list(results.items())[1]

        learning_curves_df = pd.DataFrame({
            "train_error": train_set[1]["error"],
            "train_auc": train_set[1]["auc"],
            "valid_error": valid_set[1]["error"],
            "valid_auc": valid_set[1]["auc"]})

        plt.clf()
        _, ax = plt.subplots()
        plt.xlabel('# training examples')
        plt.ylabel('auc')
        plt.title('learning curve - auc')
        ax.plot(learning_curves_df.train_auc, label='train')
        ax.plot(learning_curves_df.valid_auc, label='valid')
        context.log_artifact(
            PlotArtifact("learning curve - auc", body=plt.gcf()),
            local_path=f"{plots_dest}/learning curve - auc.html")

        plt.clf()
        _, ax = plt.subplots()
        plt.xlabel('# training examples')
        plt.ylabel('error rate')
        plt.title('learning curve - error')
        ax.plot(learning_curves_df.train_error, label='train')
        ax.plot(learning_curves_df.valid_error, label='valid')
        context.log_artifact(
            PlotArtifact("learning curve - erreur", body=plt.gcf()),
            local_path=f"{plots_dest}/learning curve - erreur.html")

    (fi_plot, fi_tbl) = feature_importances(model, xtest.columns)
    mm_plots.append(fi_plot)
    mm_tables.append(fi_tbl)

    mm_plots.append(confusion_matrix(model, xtest, ytest))

    if is_multiclass:
        lb = LabelBinarizer()
        ytest_b = lb.fit_transform(ytest)

        mm_plots.append(precision_recall_multi(ytest_b, yprob, unique_labels))
        mm_plots.append(roc_multi(ytest_b, yprob, unique_labels))

        # AUC multiclass
        aucmicro = metrics.roc_auc_score(ytest_b, yprob, multi_class="ovo",
                                         average="micro")
        aucweighted = metrics.roc_auc_score(ytest_b, yprob, multi_class="ovo",
                                             average="weighted")

        mm.update({"auc-micro": aucmicro,  "auc-weighted": aucweighted})

        # others (todo - macro, micro...)
        f1 = metrics.f1_score(ytest, ypred, average="micro")
        ps = metrics.precision_score(ytest, ypred, average="micro")
        rs = metrics.recall_score(ytest, ypred, average="micro")
        mm.update({
            "f1-score": f1,
            "precision_score": ps,
            "recall_score": rs})

    else:
        yprob_pos = yprob[:, 1]

        mm_plots.append(roc_bin(ytest, yprob_pos))
        mm_plots.append(precision_recall_bin(model, xtest, ytest, yprob_pos))

        rocauc = metrics.roc_auc_score(ytest, yprob_pos)
        brier_score = metrics.brier_score_loss(ytest, yprob_pos,
                                               pos_label=ytest.max())
        f1 = metrics.f1_score(ytest, ypred)
        ps = metrics.precision_score(ytest, ypred)
        rs = metrics.recall_score(ytest, ypred)
        mm.update({
            "rocauc": rocauc,
            "brier_score": brier_score,
            "f1-score": f1,
            "precision_score": ps,
            "recall_score": rs})

    # return all model metrics and plots
    mm.update({"plots": mm_plots, "tables": mm_tables})

    return mm
