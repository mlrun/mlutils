from mlrun.datastore import DataItem
from sklearn.model_selection import train_test_split


def get_sample(src: DataItem, sample: int, label: str, reader=None):
    """generate data sample to be split (candidate for mlrun)

    Returns features matrix and header (x), and labels (y)
    :param src:    data artifact
    :param sample: sample size from data source, use negative
                   integers to sample randomly, positive to
                   sample consecutively from the first row
    :param label:  label column title
    """
    table = src.as_df()

    # get sample
    if (sample == -1) or (sample >= 1):
        # get all rows, or contiguous sample starting at row 1.
        raw = table.dropna()
        labels = raw.pop(label)
        raw = raw.iloc[:sample, :]
        labels = labels.iloc[:sample]
    else:
        # grab a random sample
        raw = table.dropna().sample(sample * -1)
        labels = raw.pop(label)

    return raw, labels, raw.columns.values


def get_splits(
    raw,
    labels,
    n_ways: int = 3,
    test_size: float = 0.15,
    valid_size: float = 0.30,
    label_names: list = ["labels"],
    random_state: int = 1
):
    """generate train and test sets (candidate for mlrun)
    cross validation:
    1. cut out a test set
    2a. use the training set in a cross validation scheme, or
    2b. make another split to generate a validation set

    2 parts (n_ways=2): train and test set only
    3 parts (n_ways=3): train, validation and test set

    :param raw:            dataframe or numpy array of raw features
    :param labels:         dataframe or numpy array of raw labels
    :param n_ways:         (3) split data into 2 or 3 parts
    :param test_size:      proportion of raw data to set asid as test data
    :param valid_size:     proportion of remaining data to be set as validation
    :param label_names:         label names
    :param random_state:   (1) random number seed
    """
    x, xte, y, yte = train_test_split(raw, labels, test_size=test_size,
                                      random_state=random_state)
    if n_ways == 2:
        return (x, y), (xte, yte), None, None
    elif n_ways == 3:
        xtr, xva, ytr, yva = train_test_split(x, y, train_size=valid_size,
                                              random_state=random_state)
        return (xtr, ytr), (xva, yva), (xte, yte), None
    else:
        raise Exception("n_ways must be in the range [2,3]")
