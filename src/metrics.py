import pandas as pd
from typing import List, Dict
from logger import setup_custom_logger

"""
Author: Fernando Gallego
Affiliation: Researcher at the Computational Intelligence (ICB) Group, University of Málaga
"""

logger = setup_custom_logger("topk_accuracy_calculator")


def calculate_topk_accuracy(
    df: pd.DataFrame,
    topk_values: List[int]
) -> Dict[int, float]:
    """
    Calculate the Top-k accuracy for each value of k in `topk_values`.

    Parameters:
        df (pd.DataFrame):
            DataFrame containing the columns 'code' (true code) and 'codes' (predicted codes).
        topk_values (List[int]):
            List of k values to calculate the Top-k accuracy.

    Returns:
        Dict[int, float]: Dictionary with k values as keys and accuracies as values.

    Example:
        Input DataFrame:
        | code    | codes                 |
        |---------|-----------------------|
        | A1234   | [A1234, B5678, C9101] |
        | X9876   | [X9876, A1234, B5678] |

        Output:
        {1: 0.5, 2: 1.0, 3: 1.0}  # For top-1, top-2, top-3
    """
    logger.info("Calculating Top-k accuracy...")
    logger.info(f"DataFrame size: {len(df)}, Top-k values: {topk_values}")

    # Initialize a dictionary to track accuracies for each k
    topk_accuracies = {k: 0 for k in topk_values}

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        true_code = row['code']
        predicted_codes = row['codes']

        # Ensure uniqueness of predicted codes
        seen = set()
        unique_candidates = [x for x in predicted_codes if not (x in seen or seen.add(x))]

        # Check if true code is within the top-k candidates
        for k in topk_values:
            if true_code in unique_candidates[:k]:
                topk_accuracies[k] += 1

    # Normalize by the total number of rows
    total_rows = len(df)
    if total_rows == 0:
        logger.warning("DataFrame is empty. Returning zero accuracies.")
        return {k: 0.0 for k in topk_values}

    for k in topk_values:
        topk_accuracies[k] = topk_accuracies[k] / total_rows
        logger.info(f"Top-{k} accuracy: {topk_accuracies[k]:.4f}")

    logger.info("Top-k accuracy calculation complete.")
    return topk_accuracies
