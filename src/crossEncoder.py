import pandas as pd
from tqdm import tqdm
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator, CEBinaryClassificationEvaluator
from sklearn.model_selection import train_test_split
from typing import List, Dict, Optional
from logger import setup_custom_logger

"""
Author: Fernando Gallego, Guillermo López García & Luis Gasco Sánchez
Affiliation: Researcher at the Computational Intelligence (ICB) Group, University of Málaga & Barcelona Supercomputing Center (BSC)
"""

# Logger setup
logger = setup_custom_logger("cross_encoder_reranker")


class CrossEncoderReranker(CrossEncoder):
    """
    Custom CrossEncoder class for entity linking and reranking tasks.
    """

    def __init__(
        self,
        model_name: str,
        model_type: str = "mask",
        max_seq_length: int = 256
    ):
        """
        Initialize the CrossEncoderReranker with the given model.

        Args:
            model_name (str): Name or path of the model.
            model_type (str): Type of the model ("mask" for binary, others for regression). Defaults to "mask".
            max_seq_length (int): Maximum sequence length for input. Defaults to 256.
        """
        super().__init__(
            model_name,
            max_length=max_seq_length if model_type == "mask" else None,
            num_labels=1 if model_type == "mask" else None
        )
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs!")
            self = DataParallel(self)

    def prepare_triplets(
        self,
        triplets_df: pd.DataFrame
    ) -> List[InputExample]:
        """
        Prepare triplets for training.

        Args:
            triplets_df (pd.DataFrame): DataFrame containing triplets with columns ["anchor", "positive", "negative"].

        Returns:
            List[InputExample]: List of InputExample objects for training.
        """
        logger.info("Preparing triplets for training...")
        positive_triplets = triplets_df[["anchor", "positive"]].drop_duplicates().reset_index(drop=True)
        positive_triplets["label"] = 1
        positive_triplets.columns = ["anchor", "descriptor", "label"]

        negative_triplets = triplets_df[["anchor", "negative"]].drop_duplicates().reset_index(drop=True)
        negative_triplets["label"] = 0
        negative_triplets.columns = ["anchor", "descriptor", "label"]

        triplets_df_prepared = pd.concat([positive_triplets, negative_triplets])
        triplets_samples = [
            InputExample(texts=[row["anchor"], row["descriptor"]], label=row["label"])
            for _, row in triplets_df_prepared.iterrows()
        ]

        logger.info(f"Prepared {len(triplets_samples)} triplets.")
        return triplets_samples

    def transform_triplets_rankingeval(
        self,
        df_triplets: List[InputExample]
    ) -> List[Dict[str, List[str]]]:
        """
        Transform triplets into a format suitable for CERerankingEvaluator.

        Args:
            df_triplets (List[InputExample]): List of InputExample objects.

        Returns:
            List[Dict[str, List[str]]]: Transformed triplets for ranking evaluation.
        """
        logger.info("Transforming triplets for ranking evaluation...")
        dev_samples_dict = {}
        for sample in df_triplets:
            key = sample.texts[0]
            if key not in dev_samples_dict:
                dev_samples_dict[key] = {"query": key, "positive": set(), "negative": set()}
            dev_samples_dict[key]["positive" if sample.label == 1 else "negative"].add(sample.texts[1])

        transformed_samples = [
            {"query": key, "positive": list(value["positive"]), "negative": list(value["negative"])}
            for key, value in dev_samples_dict.items()
        ]
        logger.info("Transformation complete.")
        return transformed_samples

    def train(
        self,
        df_hard_triplets: pd.DataFrame,
        output_path: str,
        batch_size: int,
        epochs: int,
        evaluator_type: Optional[str] = None,
        optimizer_parameters: Dict[str, float] = {"lr": 1e-5},
        weight_decay: float = 0.01,
        evaluation_steps: int = 10000,
        save_best_model: bool = True,
        test_size: Optional[float] = None
    ) -> None:
        """
        Train the CrossEncoder model.

        Args:
            df_hard_triplets (pd.DataFrame): DataFrame containing hard triplets.
            output_path (str): Path to save the trained model.
            batch_size (int): Batch size for training.
            epochs (int): Number of training epochs.
            evaluator_type (Optional[str]): Type of evaluator to use ("BinaryClassificationEvaluator" or "CERankingEvaluator"). Defaults to None.
            optimizer_parameters (Dict[str, float]): Optimizer parameters. Defaults to {"lr": 1e-5}.
            weight_decay (float): Weight decay for optimizer. Defaults to 0.01.
            evaluation_steps (int): Steps between evaluations. Defaults to 10000.
            save_best_model (bool): Whether to save the best model. Defaults to True.
            test_size (Optional[float]): Fraction of data to use for testing. Defaults to None.
        """
        logger.info("Starting training process...")
        if test_size:
            train_samples, dev_samples = train_test_split(
                df_hard_triplets, test_size=test_size, stratify=df_hard_triplets['anchor']
            )
        else:
            train_samples = df_hard_triplets
            dev_samples = None

        train_dataloader = DataLoader(
            self.prepare_triplets(train_samples),
            shuffle=True,
            batch_size=batch_size
        )

        evaluator = None
        if evaluator_type == "BinaryClassificationEvaluator":
            evaluator = CEBinaryClassificationEvaluator.from_input_examples(
                self.prepare_triplets(dev_samples), name='dev'
            )
        elif evaluator_type == "CERankingEvaluator":
            evaluator = CERerankingEvaluator(
                self.transform_triplets_rankingeval(dev_samples), name='dev'
            )

        self.fit(
            train_dataloader=train_dataloader,
            evaluator=evaluator,
            epochs=epochs,
            optimizer_params=optimizer_parameters,
            weight_decay=weight_decay,
            evaluation_steps=evaluation_steps,
            warmup_steps=int(len(train_samples) / batch_size * epochs * 0.1),
            output_path=output_path,
            save_best_model=save_best_model
        )
        logger.info("Training complete. Model saved to %s.", output_path)

    def rerank_candidates(
        self,
        df: pd.DataFrame,
        entity_col: str,
        candidates_col: str,
        codes_col: str
    ) -> pd.DataFrame:
        """
        Rerank candidates for each entity in the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing entity, candidates, and codes.
            entity_col (str): Column name for the entity.
            candidates_col (str): Column name for the candidates.
            codes_col (str): Column name for the codes.

        Returns:
            pd.DataFrame: DataFrame with reranked candidates and codes.
        """
        logger.info("Reranking candidates...")
        if any(col not in df for col in [entity_col, candidates_col, codes_col]):
            raise ValueError("Specified columns not found in the DataFrame")

        for index in tqdm(df.index, desc="Reranking candidates"):
            entity = df.at[index, entity_col]
            candidates = df.at[index, candidates_col]
            scores = self.predict([[entity, candidate] for candidate in candidates])
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            df.at[index, candidates_col] = [candidates[i] for i in sorted_indices]
            df.at[index, codes_col] = [df.at[index, codes_col][i] for i in sorted_indices]

        logger.info("Reranking complete.")
        return df
