import numpy as np


def normalize_labels(labels):
    labels_np = np.array(labels)
    return labels_np / labels_np.sum()
