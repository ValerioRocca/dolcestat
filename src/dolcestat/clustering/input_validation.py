def validate_init_strategy(init_strategy):
    valid = {"random", "kmeans++"}
    if init_strategy not in valid:
        raise ValueError(
            f"init_strategy must be one of {sorted(valid)}, got '{init_strategy}'."
        )


def validate_flavor(flavor):
    valid = {"kmeans", "kmedians"}
    if flavor not in valid:
        raise ValueError(f"flavor must be one of {sorted(valid)}, got '{flavor}'.")


def validate_max_iters(max_iters):
    if max_iters < 1:
        raise ValueError(f"max_iters must be a positive integer, got {max_iters}.")


def validate_linkage(linkage):
    valid = {"single", "complete", "average", "ward"}
    if linkage not in valid:
        raise ValueError(f"linkage must be one of {sorted(valid)}, got '{linkage}'.")
