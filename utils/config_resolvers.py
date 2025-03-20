from omegaconf import OmegaConf
from functools import reduce
import operator

def sum_resolver(*args: int) -> int:
    """Sum resolver for Hydra configs that accepts integers and returns their sum."""
    assert all(isinstance(x, int) for x in args), "All arguments must be integers"
    return sum(args)

# Register sum resolver with the new syntax
OmegaConf.register_new_resolver("sum", sum_resolver) 