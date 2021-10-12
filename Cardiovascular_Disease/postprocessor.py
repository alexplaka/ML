import pandas as pd


def find_mislabels(y_test, model_preds):
    """
    Find all mislabels given the predictions of a given estimator/model.
    Find the false positive and negatives.
    RETURN: list of indices of all mislabeled, false positive, and false negative samples

    """

    y_pred = pd.Series(model_preds, index=y_test.index)
    diffs = y_pred - y_test
    mislabels = diffs[diffs != 0]  # All mislabeled samples
    fp = mislabels[mislabels == 1]  # False positives
    fn = mislabels[mislabels == -1]  # False negatives

    return list(mislabels.index), list(fp.index), list(fn.index)


def find_common_mislabels(*args):
    """
    Find mislabeled predictions that are common to several estimators/models.

    Note: This function can be used generally to find common entries in iterables
    by converting them to sets and finding their intersections.

    """

    common = set()
    for i in range(len(args) - 1):
        common.update(set(args[i]).intersection(set(args[i + 1])))

    return common


def model_group_mislabels(y_test, *preds):
    """
    Find the intersection of the set of all mislabels, false positives, and false negatives
    from a group or collection of estimators/models.

    PARAMETER preds: arrays/lists containing the predictions of the estimators/models.
    """

    mislabels, fp, fn = [], [], []

    for pred in preds:
        m, p, n = find_mislabels(y_test, pred)
        mislabels.append(m)
        fp.append(p)
        fn.append(n)

    common_mislabels = find_common_mislabels(*mislabels)
    common_fp = find_common_mislabels(*fp)
    common_fn = find_common_mislabels(*fn)

    return common_mislabels, common_fp, common_fn
