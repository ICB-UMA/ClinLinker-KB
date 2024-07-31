import pandas as pd

"""
Author: Fernando Gallego
Affiliation: Researcher at the Computational Intelligence (ICB) Group, University of Málaga
"""

def calculate_topk_accuracy(df, topk_values):
    """
    Calculate the Top-k accuracy for each value of k in topk_values.
    
    :param df: DataFrame containing the columns 'code' and 'codes'.
    :param topk_values: List of k values to calculate the Top-k accuracy.
    :return: Dictionary with k values as keys and accuracies as values.
    """
    topk_accuracies = {k: 0 for k in topk_values}

    for index, row in df.iterrows():
        true_code = row['code']
        predicted_codes = row['codes']
        seen = set()
        unique_candidates = [x for x in predicted_codes if not (x in seen or seen.add(x))]

        for k in topk_values:
            if true_code in unique_candidates[:k]:
                topk_accuracies[k] += 1

    total_rows = len(df)
    for k in topk_values:
        topk_accuracies[k] = topk_accuracies[k] / total_rows

    return topk_accuracies

