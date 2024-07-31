import pandas as pd

"""
Author: Fernando Gallego
Affiliation: Researcher at the Computational Intelligence (ICB) Group, University of Málaga
"""

class TripletsGeneration:
    """
    A base class for generating triplets.

    Attributes:
        df (pandas.DataFrame): DataFrame with necessary columns.
    """

    def __init__(self, df):
        """
        Initialize the TripletsGeneration class with a DataFrame.
        
        Args:
            df (pandas.DataFrame): DataFrame with necessary columns.
        """
        self.df = df

    def generate_triplets(self):
        """
        Placeholder method to be overridden by subclasses.
        
        Raises:
            NotImplementedError: This method should be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

class TopHardTriplets(TripletsGeneration):
    """
    Subclass of TripletsGeneration that generates top-hard triplets.
    """

    def generate_triplets(self, num_negatives=None):
        """
        Generates top-hard triplets.

        Args:
            num_negatives (int, optional): Number of negative samples to consider. Defaults to None.

        Returns:
            pandas.DataFrame: DataFrame containing anchor, positive, and negative columns.
        """
        results = []
        for index, row in self.df.iterrows():
            term, correct_code = row['term'], row['code']
            candidate_codes, candidate_texts = row['codes'], row['candidates'] 

            if correct_code in candidate_codes:
                positive_index = candidate_codes.index(correct_code)
                positive_text = candidate_texts[positive_index]

                negatives = candidate_texts[:num_negatives] if num_negatives else candidate_texts[:positive_index]
                for neg_text in negatives:
                    results.append((term, positive_text, neg_text))

        return pd.DataFrame(results, columns=["anchor", "positive", "negative"])

class SimilarityHardTriplets(TripletsGeneration):
    """
    Subclass of TripletsGeneration that generates triplets based on text similarity.
    """

    def generate_triplets(self, similarity_threshold):
        """
        Generates triplets based on text similarity.

        Args:
            similarity_threshold (float): Similarity threshold.

        Returns:
            pandas.DataFrame: DataFrame containing anchor, positive, and negative columns.
        """
        results = []
        for index, row in self.df.iterrows():
            term, correct_code = row['term'], row['code']
            candidate_texts, candidate_similarities = row['candidates'], row['similarities']

            if correct_code in row['codes']:
                positive_index = row['codes'].index(correct_code)
                positive_text = candidate_texts[positive_index]

                negative_indices = [i for i, sim in enumerate(candidate_similarities) if sim > similarity_threshold and i != positive_index]
                for i in negative_indices:
                    results.append((term, positive_text, candidate_texts[i]))

        return pd.DataFrame(results, columns=["anchor", "positive", "negative"])