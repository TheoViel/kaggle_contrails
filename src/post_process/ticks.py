import numpy as np


def my_assignment(mat):
    """
    Performs assignment of rows and columns based on the minimum values in the given matrix.

    Args:
        mat (np.ndarray): The input matrix.

    Returns:
        List[int], List[int]: Lists of row indices and column indices representing the assignment.
    """
    row_ind, col_ind = [], []
    for i in range(np.min(mat.shape)):
        row, col = np.unravel_index(np.argmin(mat), mat.shape)
        mat[row] = np.inf
        mat[:, col] = np.inf
        row_ind.append(row)
        col_ind.append(col)

    return row_ind, col_ind


def assign(ticks, labels, tol=2, mode="x"):
    """
    Assign labels to ticks based on their proximity.

    Args:
        ticks (numpy.ndarray): Array of tick coordinates.
        labels (numpy.ndarray): Array of label coordinates.
        tol (int): Tolerance value for assigning labels to ticks.
        mode (str): Mode indicating whether to assign labels along the x-axis ("x") or y-axis ("y").

    Returns:
        numpy.ndarray: Assigned tick coordinates.
        numpy.ndarray: Assigned label coordinates.
    """
    if mode == "x":
        labels_x, labels_y = (labels[:, 0] + labels[:, 2]) / 2, labels[:, 1]
    else:
        labels_x, labels_y = labels[:, 2], (labels[:, 1] + labels[:, 3]) / 2

    labels_xy = np.stack([labels_x, labels_y], -1)

    ticks_x, ticks_y = (ticks[:, 0] + ticks[:, 2]) / 2, (ticks[:, 1] + ticks[:, 3]) / 2
    ticks_xy = np.stack([ticks_x, ticks_y], -1)

    cost_matrix = np.sqrt(((ticks_xy[:, None] - labels_xy[None]) ** 2).sum(-1))

    #     print(np.min(cost_matrix))
    if mode == "x":  # penalize y_label < y_tick
        cost_matrix += (
            ((ticks_y[:, None] - labels_y[None]) > 0) * np.min(cost_matrix) * tol
        )
    else:  # penalize x_tick < x_label
        cost_matrix += (
            ((ticks_x[:, None] - labels_x[None]) < 0) * np.min(cost_matrix) * tol
        )

    row_ind, col_ind = my_assignment(cost_matrix.copy())

    ticks_assigned, labels_assigned = [], []
    for tick_idx, label_idx in zip(row_ind, col_ind):
        if cost_matrix[tick_idx, label_idx] < max(tol * 5, tol * np.min(cost_matrix)):
            ticks_assigned.append(ticks[tick_idx])
            labels_assigned.append(labels[label_idx])

    # Fix outlier too close
    if len(ticks_assigned) <= 3:
        error_value = np.min(cost_matrix)
        cost_matrix = np.where(cost_matrix < error_value * 2, 100000, cost_matrix)

        ticks_assigned_fixed, labels_assigned_fixed = [], []
        for tick_idx, label_idx in zip(row_ind, col_ind):
            if cost_matrix[tick_idx, label_idx] < max(tol * 5, tol * np.min(cost_matrix)):
                ticks_assigned_fixed.append(ticks[tick_idx])
                labels_assigned_fixed.append(labels[label_idx])

        if len(ticks_assigned) < len(ticks_assigned_fixed):
            ticks_assigned = ticks_assigned_fixed
            labels_assigned = labels_assigned_fixed

    return np.array(ticks_assigned), np.array(labels_assigned)


def restrict_on_line(preds, margin=5, cat=False):
    """
    Restrict the predicted ticks on the x-axis and y-axis to be aligned along a line.

    Args:
        preds (list): List of predicted elements containing chart, x-axis ticks, y-axis ticks, and points.
        margin (int): Margin value for considering ticks as aligned along a line.
        cat (bool): Flag indicating whether to concatenate x-axis and y-axis labels and ticks.

    Returns:
        list: List of restricted predicted elements.
    """
    try:
        graph = preds[0][0]
        x_axis, y_axis = graph[0], graph[3]
    except Exception:
        x_axis, y_axis = 0, 0

    ticks = preds[2]
    ticks_x, ticks_y = (ticks[:, 0] + ticks[:, 2]) / 2, (ticks[:, 1] + ticks[:, 3]) / 2

    dists_x = ticks_x - x_axis
    dists_y = ticks_y - y_axis

    best_x = dists_x[np.argmax([(np.abs(dists_x - d) < margin).sum() for d in dists_x])]
    best_y = dists_y[np.argmax([(np.abs(dists_y - d) < margin).sum() for d in dists_y])]

    y_ticks = ticks[np.abs(dists_x - best_x) < margin]  # similar x
    x_ticks = ticks[np.abs(dists_y - best_y) < margin]  # similar y

    # Pair with labels
    labels = preds[1]

    x_ticks, x_labels = assign(x_ticks.copy(), labels.copy())
    y_ticks, y_labels = assign(y_ticks.copy(), labels.copy(), mode="y")

    # Reorder
    order_x = np.argsort(x_ticks[:, 0])
    x_ticks = x_ticks[order_x]
    x_labels = x_labels[order_x]

    order_y = np.argsort(y_ticks[:, 1])[::-1]
    y_ticks = y_ticks[order_y]
    y_labels = y_labels[order_y]

    if not cat:
        return [preds[0], x_labels, y_labels, x_ticks, y_ticks, preds[3]]

    labels = np.unique(np.concatenate([x_labels, y_labels]), axis=0)
    ticks = np.unique(np.concatenate([x_ticks, y_ticks]), axis=0)

    return [preds[0], labels, ticks, preds[3]]
