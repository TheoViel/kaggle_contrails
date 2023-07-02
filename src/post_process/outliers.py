import numpy as np
from scipy.stats import spearmanr, pearsonr


def find_outliers(x_train, values, verbose=0, corr="spearman", th=0.99):
    """
    Finds outliers in a dataset based on their correlation with a target variable.

    Args:
        x_train (list or numpy.array): Training data features.
        values (list or numpy.array): Target variable values.
        verbose (int): Verbosity level. Set to 1 to print removal information. Defaults to 0.
        corr (str): Correlation type. Defaults to "spearman".
        th (float): Threshold for outlier correlation. Defaults to 0.99.

    Returns:
        list: List of indices corresponding to the outliers found.
    """
    corr_fct = spearmanr if corr == "spearman" else pearsonr

    corr_start = np.abs(corr_fct(x_train, values).statistic)
    if corr_start > th:
        return []

    # One outlier
    if len(x_train) > 2:
        for i in range(len(x_train)):
            x_train_ = [x for j, x in enumerate(x_train) if j != i]
            values_ = [v for j, v in enumerate(values) if j != i]
            corr = np.abs(corr_fct(x_train_, values_).statistic)

            if corr > th:
                if verbose:
                    print(f"Remove {i}")
                return [i]

    # Two outliers
    if len(x_train) > 3:
        for i in range(len(x_train)):
            for i2 in range(i):
                x_train_ = [x for j, x in enumerate(x_train) if (j != i and j != i2)]
                values_ = [v for j, v in enumerate(values) if (j != i and j != i2)]
                corr = np.abs(corr_fct(x_train_, values_).statistic)

                if corr > th:
                    if verbose:
                        print(f"Remove {i}, {i2}")
                    return [i, i2]

    return []


def longest_increasing_subset(lst):
    """
    Finds the longest increasing subsequence in a given list.

    Args:
        lst (list): Input list.

    Returns:
        list: Longest increasing subsequence.
    """
    n = len(lst)
    if n == 0:
        return []

    # Initialize the lengths and previous indices
    lengths = [1] * n
    previous_indices = [-1] * n

    # Iterate over the list and update the lengths and previous indices
    for i in range(1, n):
        for j in range(i):
            if lst[i] > lst[j] and lengths[i] < lengths[j] + 1:
                lengths[i] = lengths[j] + 1
                previous_indices[i] = j

    # Find the index of the longest increasing subsequence
    max_length_index = max(range(n), key=lambda x: lengths[x])

    # Reconstruct the longest increasing subsequence
    result = []
    while max_length_index != -1:
        result.append(lst[max_length_index])
        max_length_index = previous_indices[max_length_index]

    return result[::-1]


def find_outliers_order(values, verbose=0):
    """
    Finds outliers in a list of values based on their order.

    Args:
        values (list or numpy.ndarray): Input list of values.
        verbose (int, optional): Verbosity level (0: no output, 1: output outliers). Defaults to 0.

    Returns:
        list: List of outlier indices.
    """
    ref = np.arange(len(values))
    sort = np.argsort(values)

    # Correct order
    if (ref == sort).all() or (ref[::-1] == sort).all():
        return []

    longest_inc = longest_increasing_subset(sort)
    longest_dec = longest_increasing_subset(sort[::-1])

    #     print(longest_inc, longest_dec)

    if len(longest_inc) >= len(longest_dec):
        return [i for i in sort if i not in longest_inc]
    else:
        return [i for i in sort if i not in longest_dec]
