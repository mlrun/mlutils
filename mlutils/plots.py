# Copyright 2020 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns
from mlrun.artifacts import PlotArtifact, TableArtifact
import pandas as pd


def gcf_clear(plt):
    """Utility to clear matplotlib figure
    Run this inside every plot method before calling any matplotlib
    methods
    :param plot:    matloblib figure object
    """
    plt.cla()
    plt.clf()
    plt.close()


def feature_importances(model, header):
    """Display estimated feature importances
    Only works for models with attribute 'feature_importances_`
    :param model:       fitted model
    :param header:      feature labels
    """
    if not hasattr(model, "feature_importances_"):
        raise Exception(
            "feature importaces are only available for some models")

    # create a feature importance table with desired labels
    zipped = zip(model.feature_importances_, header)
    feature_imp = pd.DataFrame(sorted(zipped), columns=["freq", "feature"]).sort_values(
        by="freq", ascending=False
    )

    plt.clf() #gcf_clear(plt)
    plt.figure(figsize=(20, 10))
    sns.barplot(x="freq", y="feature", data=feature_imp)
    plt.title("features")
    plt.tight_layout()

    return (PlotArtifact("feature-importances", body=plt.gcf()),
            TableArtifact("feature-importances-tbl", df=feature_imp))
    

def learning_curves(model):
    """model class dependent
    
    WIP
    
    get training history plots for xgboost, lightgbm
    
    returns list of PlotArtifacts, can be empty if no history
    is found
    """
    plots = []
    
    # do this here and not in the call to learning_curve plots,
    # this is default approach for xgboost and lightgbm
    if hasattr(model, "evals_result"):
        results = model.evals_result()
        train_set = list(results.items())[0]
        valid_set = list(results.items())[1]

        learning_curves = pd.DataFrame({
            "train_error" : train_set[1]["error"],
            "train_auc" : train_set[1]["auc"],
            "valid_error" : valid_set[1]["error"],
            "valid_auc" : valid_set[1]["auc"]})

        plt.clf() #gcf_clear(plt)
        fig, ax = plt.subplots()
        plt.xlabel('# training examples')
        plt.ylabel('auc')
        plt.title('learning curve - auc')
        ax.plot(learning_curves.train_auc, label='train')
        ax.plot(learning_curves.valid_auc, label='valid')
        legend = ax.legend(loc='lower left')
        plots.append(PlotArtifact("learning curve - auc",
                                  body=plt.gcf()))

        plt.clf() #gcf_clear(plt)
        fig, ax = plt.subplots()
        plt.xlabel('# training examples')
        plt.ylabel('error rate')
        plt.title('learning curve - error')
        ax.plot(learning_curves.train_error, label='train')
        ax.plot(learning_curves.valid_error, label='valid')
        legend = ax.legend(loc='lower left')
        plots.append(PlotArtifact("learning curve - taoot",
                                  body=plt.gcf()))
    # elif some other model history api...
        
    return plots

def confusion_matrix(model, xtest, ytest):
    cmd = metrics.plot_confusion_matrix(
        model, xtest, ytest, normalize='all', values_format='.2g', cmap=plt.cm.Blues)
    # for now only 1, add different views to this array for display in UI
    return PlotArtifact("confusion-matrix-normalized", body=cmd.figure_)
    
def precision_recall_multi(ytest_b, yprob, labels, scoring="micro"):
    """
    """
    n_classes = len(labels)

    precision = dict()
    recall = dict()
    avg_prec = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = metrics.precision_recall_curve(ytest_b[:, i],
                                                                    yprob[:, i])
        avg_prec[i] = metrics.average_precision_score(
            ytest_b[:, i], yprob[:, i])
    precision["micro"], recall["micro"], _ = metrics.precision_recall_curve(ytest_b.ravel(),
                                                                            yprob.ravel())
    avg_prec["micro"] = metrics.average_precision_score(
        ytest_b, yprob, average="micro")
    ap_micro = avg_prec["micro"]
    model_metrics.update({'precision-micro-avg-classes': ap_micro})

    gcf_clear(plt)
    colors = cycle(['navy', 'turquoise', 'darkorange',
                    'cornflowerblue', 'teal'])
    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=10)
    lines.append(l)
    labels.append(
        f'micro-average precision-recall (area = {ap_micro:0.2f})')

    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append(
            f'precision-recall for class {i} (area = {avg_prec[i]:0.2f})')

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('precision recall - multiclass')
    plt.legend(lines, labels, loc=(0, -.41), prop=dict(size=10))

    return PlotArtifact("precision-recall-multiclass", body=plt.gcf())

    
def roc_multi(ytest_b, yprob, labels):
    """
    """
    n_classes = len(labels)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(ytest_b[:, i], yprob[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(
        ytest_b.ravel(), yprob.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    #gcf_clear(plt)
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
                label='micro-average ROC curve (area = {0:0.2f})'
                    ''.format(roc_auc["micro"]),
                color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
                label='macro-average ROC curve (area = {0:0.2f})'
                    ''.format(roc_auc["macro"]),
                color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label='ROC curve of class {0} (area = {1:0.2f})'
                    ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('receiver operating characteristic - multiclass')
    plt.legend(loc=(0, -.68), prop=dict(size=10))
    
    return PlotArtifact("roc-multiclass", body=plt.gcf())
    
def roc_bin(ytest, yprob):
    """
    """
    # ROC plot
    #gcf_clear(plt)
    fpr, tpr, _ = metrics.roc_curve(ytest, yprob)
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='a label')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('roc curve')
    plt.legend(loc='best')
    
    return PlotArtifact("roc-binary", body=plt.gcf())

def precision_recall_bin(model, xtest, ytest, yprob):
    """
    """
    # precision-recall
    #gcf_clear(plt)
    disp = metrics.plot_precision_recall_curve(model, xtest, ytest)
    disp.ax_.set_title(
            f'precision recall: AP={metrics.average_precision_score(ytest, yprob):0.2f}')
    
    return PlotArtifact("precision-recall-binary", body=disp.figure_)
