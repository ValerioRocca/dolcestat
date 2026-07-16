class Node:
    def __init__(self, X, y, depth):
        self.X = X
        self.y = y
        self.depth = depth
        self.is_leaf = True
        self.split_feature = None
        self.split_threshold = None
        self.is_categorical_split = False
        self.left = None
        self.right = None
        self.prediction = None
