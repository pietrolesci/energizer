from energizer.active_learning import acquisition_functions, clustering_utilities
from energizer.registries import Registry

SCORING_FUNCTIONS = Registry()
SCORING_FUNCTIONS.register_functions(acquisition_functions)


CLUSTERING_FUNCTIONS = Registry()
CLUSTERING_FUNCTIONS.register_functions(clustering_utilities, filter_fn=lambda k, _: not k.startswith("_"))
