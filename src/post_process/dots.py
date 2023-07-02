import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


def cluster_on_x(dots, w, plot=False):
    """
    Clusters the given dots based on their x-coordinates.

    Args:
        dots (np.ndarray): An array of shape (N, 4) containing the bounding box coordinates of the dots.
        w (float): The width of the image.
        plot (bool, optional): Whether to plot the clusters. Defaults to False.

    Returns:
        np.ndarray: X-coordinates of the cluster centers and
        Dict[int, int] Dictionary mapping cluster labels to the count of dots in each cluster.
    """
    xs = (dots[:, 0] + dots[:, 2]) / 2
    ys = (dots[:, 1] + dots[:, 3]) / 2

    dbscan = DBSCAN(min_samples=1, eps=0.01 * w)
    dbscan.fit(xs[:, None])
    labels = dbscan.labels_

    centers = []

    for lab in np.unique(labels):
        centers.append(xs[labels == lab].mean())

        if plot:
            plt.scatter(
                xs[labels == lab],
                -ys[labels == lab],
                label=f"Cluster {lab}",
            )
    if plot:
        plt.legend()
        plt.show()

    labels = np.array(labels)
    centers = np.array(centers)
    clusters_y = [np.sort(ys[labels == i])[::-1] for i in np.unique(labels)]

    first = np.median([c[0] for c in clusters_y])
    second = np.median([c[1] for c in clusters_y if len(c) > 1])
    third = np.median([c[2] for c in clusters_y if len(c) > 2])

    if len([c[1] for c in clusters_y if len(c) > 2]) > 2:
        delta = (first - third) / 2
    else:
        delta = (first - second)

    counts = [np.round((first - c.min()) / (delta) + 1) for c in clusters_y]
    clusters = dict(zip(np.unique(labels), counts))

    return centers, clusters  # Counter(labels)


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
        value = mat[row, col]
        if value < 20:
            mat[row] = np.inf
            mat[:, col] = np.inf
            row_ind.append(row)
            col_ind.append(col)

    return row_ind, col_ind


def assign_dots(labels, centers, tol=10, retrieve_missing=False, verbose=0):
    """
    Assigns dots to labels based on their proximity to the centers.

    Args:
        labels (np.ndarray): An array of label coordinates.
        centers (np.ndarray): An array of center coordinates.
        tol (int): Tolerance value for assigning dots to labels. Defaults to 10.
        retrieve_missing (bool): Flag indicating whether to retrieve missing dots. Defaults to False.
        verbose (int): Verbosity level. Defaults to 0.

    Returns:
        Tuple[Dict[int, int], np.ndarray]: A tuple containing a dictionary mapping dot indices to label
                                           indices and an array of retrieved dots.
    """
    labels_x = (labels[:, 0] + labels[:, 2]) / 2
    cost_matrix = np.abs(labels_x[:, None] - centers[None])

    row_ind, col_ind = my_assignment(cost_matrix.copy())

    mapping = dict(zip(row_ind, col_ind))

    if not retrieve_missing:
        return mapping, []

    # Unassigned dots
    unassigned = [k for k in range(len(centers)) if k not in mapping.values()]
    centers_unassigned = centers[unassigned]

    if not len(unassigned):
        return mapping, []

    yc = (
        ((labels[:, 1] + labels[:, 3]) / 2)
        .mean(0, keepdims=True)[None]
        .repeat(len(centers_unassigned), 0)
    )
    w = (
        (labels[:, 2] - labels[:, 0])
        .mean(0, keepdims=True)[None]
        .repeat(len(centers_unassigned), 0)
    )
    h = (
        (labels[:, 3] - labels[:, 1])
        .mean(0, keepdims=True)[None]
        .repeat(len(centers_unassigned), 0)
    )
    xc = centers_unassigned[:, None]

    retrieved = np.concatenate(
        [xc - w // 2, yc - h // 2, xc + w // 2, yc + h // 2], 1
    ).astype(int)

    mapping.update({len(labels) + i: k for i, k in enumerate(unassigned)})

    return mapping, retrieved


def restrict_labels_x(preds, margin=5):
    """
    Restricts the labels in the x-axis based on their proximity to a reference point on the graph.

    Args:
        preds (List): A list of predictions containing the graph, labels, and other information.
        margin (int): The margin value used to determine proximity. Defaults to 5.

    Returns:
        List: A modified list of predictions with restricted labels in the x-axis.
    """
    try:
        graph = preds[0][0]
        x_axis, y_axis = graph[0], graph[3]
    except Exception:
        x_axis, y_axis = 0, 0

    labels = preds[1]

    if not len(labels):
        return [preds[0], np.empty((0, 4)), np.empty((0, 4)), preds[3]]

    labels_x, labels_y = (labels[:, 0] + labels[:, 2]) / 2, (
        labels[:, 1] + labels[:, 3]
    ) / 2

    dists_x = labels_x - x_axis
    dists_y = labels_y - y_axis

    best_x = dists_x[np.argmax([(np.abs(dists_x - d) < margin).sum() for d in dists_x])]
    best_y = dists_y[np.argmax([(np.abs(dists_y - d) < margin).sum() for d in dists_y])]

    y_labels = labels[np.abs(dists_x - best_x) < margin]  # similar x  # noqa
    x_labels = labels[np.abs(dists_y - best_y) < margin]  # similar y

    return [preds[0], x_labels, np.empty((0, 4)), preds[3]]


def constraint_size(dots, coef=0, margin=1):
    """
    Applies constraints on the size (width and height)
    of the dots based on their median values and coefficients.

    Args:
        dots (ndarray): An array of dots represented by their bounding boxes.
        coef (float): The coefficient used to determine the size constraint. Defaults to 0.
        margin (int): The margin value added to the size constraint. Defaults to 1.

    Returns:
        ndarray: An array of dots after applying the size constraints.
    """
    ws = dots[:, 2] - dots[:, 0]
    hs = dots[:, 3] - dots[:, 1]

    median_w = np.median(ws[:10])
    median_h = np.median(hs[:10])

    dots = dots[
        (np.abs(ws - median_w) < (median_w * coef + margin))
        & (np.abs(hs - median_h) < (median_h * coef + margin))
    ]
    return dots
