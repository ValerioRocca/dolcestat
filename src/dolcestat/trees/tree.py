"""The node a fitted decision tree is made of."""


class Node:
    """One node of a fitted tree: either a split or a leaf with a prediction."""

    def __init__(self, depth, n_samples=None):
        self.depth = depth
        # Number of training rows that reached this node
        self.n_samples = n_samples
        self.is_leaf = True
        self.split_feature = None
        self.split_threshold = None
        self.is_categorical_split = False
        self.left = None
        self.right = None
        self.prediction = None
