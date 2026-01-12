import numpy as np
import itertools
from si.model_selection.cross_validate import k_fold_cross_validation

def randomized_search_cv(model, dataset, hyperparameter_grid, scoring=None, cv=3, n_iter=10):
    """
    Perform randomized search with cross-validation to find optimal hyperparameters.

    Args:
        model: The machine learning model to optimize
        dataset: Dataset containing features and labels
        hyperparameter_grid: Dictionary of hyperparameters and their possible values
        scoring: Function to evaluate model performance (optional)
        cv: Number of cross-validation folds (default: 3)
        n_iter: Number of parameter combinations to try (default: 10)

    Returns:
        Dictionary containing:
        - hyperparameters: All tried hyperparameter combinations
        - scores: Cross-validation scores for each combination
        - best_hyperparameters: Best performing hyperparameters
        - best_score: Highest cross-validation score achieved

    Raises:
        AttributeError: If model doesn't have a hyperparameter from the grid
    """
    # Validate that all hyperparameters exist in the model
    for parameter in hyperparameter_grid:
        if not hasattr(model, parameter):
            raise AttributeError(f"Model {model.__class__.__name__} has no attribute '{parameter}'")

    results = {
        'hyperparameters': [],
        'scores': [],
        'best_hyperparameters': None,
        'best_score': -np.inf
    }

    keys, values = zip(*hyperparameter_grid.items())
    all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    n_iter = min(n_iter, len(all_combinations))

    random_indices = np.random.choice(len(all_combinations), n_iter, replace=False)
    selected_combinations = [all_combinations[i] for i in random_indices]

    for combination in selected_combinations:
        for parameter, value in combination.items():
            setattr(model, parameter, value)

        scores = k_fold_cross_validation(model, dataset, scoring=scoring, cv=cv)
        mean_score = np.mean(scores)

        results['scores'].append(mean_score)
        results['hyperparameters'].append(combination)

        if mean_score > results['best_score']:
            results['best_score'] = mean_score
            results['best_hyperparameters'] = combination

    return results