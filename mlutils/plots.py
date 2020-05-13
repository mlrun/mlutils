

def plot_roc(
    context,
    y_labels,
    y_probs,
    key="roc",
    plots_dir: str = "plots",
    fmt="png",
    fpr_label: str = "false positive rate",
    tpr_label: str = "true positive rate",
    title: str = "roc curve",
    legend_loc: str = "best",
):
    """plot roc curves

    **legacy version please deprecate in functions and demos**

    :param context:      the function context
    :param y_labels:     ground truth labels, hot encoded for multiclass
    :param y_probs:      model prediction probabilities
    :param key:          ("roc") key of plot in artifact store
    :param plots_dir:    ("plots") destination folder relative path to artifact path
    :param fmt:          ("png") plot format
    :param fpr_label:    ("false positive rate") x-axis labels
    :param tpr_label:    ("true positive rate") y-axis labels
    :param title:        ("roc curve") title of plot
    :param legend_loc:   ("best") location of plot legend
    """
    # clear matplotlib current figure
    gcf_clear(plt)

    # draw 45 degree line
    plt.plot([0, 1], [0, 1], "k--")

    # labelling
    plt.xlabel(fpr_label)
    plt.ylabel(tpr_label)
    plt.title(title)
    plt.legend(loc=legend_loc)

    # single ROC or mutliple
    if y_labels.shape[1] > 1:
        # data accummulators by class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(y_labels[:, :-1].shape[1]):
            fpr[i], tpr[i], _ = metrics.roc_curve(
                y_labels[:, i], y_probs[:, i], pos_label=1
            )
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], label=f"class {i}")
    else:
        fpr, tpr, _ = metrics.roc_curve(y_labels, y_probs[:, 1], pos_label=1)
        plt.plot(fpr, tpr, label=f"positive class")

    fname = f"{plots_dir}/{key}.html"
    return context.log_artifact(PlotArtifact(key, body=plt.gcf()), local_path=fname)
