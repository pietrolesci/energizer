from energizer.active_learning import acquisition_functions
from energizer.registries import Registry

SCORING_FUNCTIONS = Registry()
SCORING_FUNCTIONS.register_functions(acquisition_functions)
