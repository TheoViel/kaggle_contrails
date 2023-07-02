def post_process_preds(preds, margin_pt=10, margin_text=30):
    """
    Applies post-processing to the predictions obtained from a model.

    Args:
        preds (list): List of predictions containing graph, texts, ticks, and points.
        margin_pt (int): Margin value for filtering points. Defaults to 10.
        margin_text (int): Margin value for filtering texts. Defaults to 30.

    Returns:
        list: Updated list of predictions after post-processing.
    """
    try:
        graph = preds[0][0]
    except Exception:
        return preds

    # Points are inside the graph
    points = preds[3]
    margin = margin_pt
    points = points[points[:, 0] > graph[0] - margin]
    points = points[points[:, 1] > graph[1] - margin]
    points = points[points[:, 2] < graph[2] + margin]
    points = points[points[:, 3] < graph[3] + margin]

    # Texts are below or left of the graph
    texts = preds[1]
    margin = margin_text
    texts = texts[
        (texts[:, 1] > graph[3] - margin)
        | (texts[:, 0] < graph[0] + margin)  # left  # bottom
    ]

    # Ticks are on the axis
    ticks = preds[2]
    return [preds[0], texts, ticks, points]


def post_process_preds_dots(preds, margin_pt=10, margin_text=30):
    """
    Applies post-processing to the predictions obtained from a model.

    Args:
        preds (list): List of predictions containing graph, texts, ticks, and points.
        margin_pt (int): Margin value for filtering points. Defaults to 10.
        margin_text (int): Margin value for filtering texts. Defaults to 30.

    Returns:
        list: Updated list of predictions after post-processing.
    """
    try:
        graph = preds[0][0]
    except Exception:
        return preds

    # Points are inside the graph
    points = preds[3]
    margin = margin_pt
#     print(graph)
    points = points[points[:, 2] > graph[0] + margin]
    points = points[points[:, 1] > graph[1] - margin]
    points = points[points[:, 2] < graph[2] + margin]
    points = points[points[:, 1] < graph[3] - margin]

    if not len(points):
        points = preds[3]

    # Texts are below & right
    texts = preds[1]
    margin = margin_text

    texts = texts[
        ((texts[:, 2] + texts[:, 0]) / 2) > (graph[0] - margin)
    ]
    texts = texts[
        ((texts[:, 3] + texts[:, 1]) / 2) > (graph[3] - margin)
    ]
    if not len(texts):
        texts = preds[1]

    ticks = preds[2]

    return [preds[0], texts, ticks, points]
