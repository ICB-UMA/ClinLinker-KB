import pandas as pd
from typing import Optional, List
from logger import setup_custom_logger

"""
Author: Fernando Gallego
Affiliation: Researcher at the Computational Intelligence (ICB) Group, University of Málaga
"""

logger = setup_custom_logger("triplets_generation")


class TripletsGeneration:
    """
    A base class for generating triplets.

    Attributes:
        df (pd.DataFrame): DataFrame with necessary columns.
    """

    def __init__(
        self,
        df: pd.DataFrame
    ):
        """
        Initialize the TripletsGeneration class with a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame with necessary columns.
        """
        self.df = df
        self._validate_dataframe()

    def _validate_dataframe(self) -> None:
        """Validate that the DataFrame contains the required columns."""
        required_columns = {'term', 'code', 'codes', 'candidates'}
        if not required_columns.issubset(self.df.columns):
            missing = required_columns - set(self.df.columns)
            raise ValueError(f"DataFrame is missing required columns: {missing}")
        logger.info("DataFrame validation successful.")

    def generate_triplets(self) -> pd.DataFrame:
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

    def generate_triplets(
        self,
        num_negatives: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generates top-hard triplets.

        Args:
            num_negatives (Optional[int]): Number of negative samples to consider. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing anchor, positive, and negative columns.
        """
        logger.info("Generating top-hard triplets...")
        results = []
        for _, row in self.df.iterrows():
            term = row['term']
            correct_code = row['code']
            candidate_codes = row['codes']
            candidate_texts = row['candidates']

            if correct_code in candidate_codes:
                positive_index = candidate_codes.index(correct_code)
                positive_text = candidate_texts[positive_index]

                negatives = (
                    candidate_texts[:num_negatives]
                    if num_negatives else candidate_texts[:positive_index]
                )
                for neg_text in negatives:
                    results.append((term, positive_text, neg_text))

        logger.info(f"Generated {len(results)} triplets.")
        return pd.DataFrame(results, columns=["anchor", "positive", "negative"])


class SimilarityHardTriplets(TripletsGeneration):
    """
    Subclass of TripletsGeneration that generates triplets based on text similarity.
    """

    def generate_triplets(
        self,
        similarity_threshold: float
    ) -> pd.DataFrame:
        """
        Generates triplets based on text similarity.

        Args:
            similarity_threshold (float): Similarity threshold.

        Returns:
            pd.DataFrame: DataFrame containing anchor, positive, and negative columns.
        """
        logger.info(f"Generating similarity-based triplets with threshold: {similarity_threshold}...")
        results = []
        for _, row in self.df.iterrows():
            term = row['term']
            correct_code = row['code']
            candidate_texts = row['candidates']
            candidate_similarities = row['similarities']

            if correct_code in row['codes']:
                positive_index = row['codes'].index(correct_code)
                positive_text = candidate_texts[positive_index]

                negative_indices = [
                    i for i, sim in enumerate(candidate_similarities)
                    if sim > similarity_threshold and i != positive_index
                ]
                for i in negative_indices:
                    results.append((term, positive_text, candidate_texts[i]))

        logger.info(f"Generated {len(results)} triplets.")
        return pd.DataFrame(results, columns=["anchor", "positive", "negative"])
