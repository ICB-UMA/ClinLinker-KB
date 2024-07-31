from sentence_transformers import CrossEncoder
from torch.utils.data import DataLoader
from sentence_transformers import InputExample
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import torch
from torch.nn import DataParallel
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator, CEBinaryClassificationEvaluator

"""
Author: Fernando Gallego, Guillermo López García & Luis Gasco Sánchez
Affiliation: Researcher at the Computational Intelligence (ICB) Group, University of Málaga & Barcelona Supercomputing Center (BSC)
"""

class CrossEncoderReranker(CrossEncoder):
    def __init__(self, model_name: str, model_type="mask", max_seq_length:int = 256):
        super().__init__(model_name, max_length=max_seq_length if model_type == "mask" else None, num_labels=1 if model_type == "mask" else None)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self = DataParallel(self)

    def prepare_triplets(self, triplets_df):
        positive_triplets = triplets_df[["anchor", "positive"]].drop_duplicates().reset_index(drop=True)
        positive_triplets["label"] = 1
        positive_triplets.columns = ["anchor","descriptor","label"]

        negative_triplets = triplets_df[["anchor", "negative"]].drop_duplicates().reset_index(drop=True)
        negative_triplets["label"] = 0
        negative_triplets.columns = ["anchor","descriptor","label"]

        triplets_df_prepared = pd.concat([positive_triplets,negative_triplets])
        triplets_samples = list()
        for sentence1, sentence2, label_id in zip(triplets_df_prepared.anchor, triplets_df_prepared.descriptor, triplets_df_prepared.label):
            triplets_samples.append(InputExample(texts = [sentence1,sentence2], label = label_id))
        
        return triplets_samples


    def transform_triplets_rankingeval(self, df_triplets):
        dev_samples_dict = {}
        for sample in df_triplets:
            key = sample.texts[0]
            if key not in dev_samples_dict:
                dev_samples_dict[key] = {"query": key, "positive": set(), "negative": set()}
            dev_samples_dict[key]["positive" if sample.label == 1 else "negative"].add(sample.texts[1])
        return [{"query": key, "positive": list(value["positive"]), "negative": list(value["negative"])} for key, value in dev_samples_dict.items()]

    def train(self, df_hard_triplets, output_path, batch_size, epochs, evaluator_type=None, optimizer_parameters={"lr":1e-5}, weight_decay=0.01, evaluation_steps=10000, save_best_model=True, test_size=None):
        if test_size:
            train_samples, dev_samples = train_test_split(df_hard_triplets, test_size=test_size, stratify=df_hard_triplets['anchor'])
        else:
            train_samples = df_hard_triplets
        train_dataloader = DataLoader(self.prepare_triplets(train_samples), shuffle=True, batch_size=batch_size)

        if evaluator_type == None:
            evaluator = None
        elif evaluator_type == "BinaryClassificationEvaluator":
            evaluator = CEBinaryClassificationEvaluator.from_input_examples(self.prepare_triplets(dev_samples), name='dev') 
        elif evaluator_type == "CERankingEvaluator":
            evaluator = CERerankingEvaluator(self.transform_triplets_rankingeval(dev_samples), name='dev')


        self.fit(train_dataloader=train_dataloader, evaluator=evaluator, epochs=epochs,
                     optimizer_params=optimizer_parameters, weight_decay=weight_decay,
                     evaluation_steps=evaluation_steps, warmup_steps=int(len(train_samples) / batch_size * epochs * 0.1),
                     output_path=output_path, save_best_model=save_best_model)
            
    def rerank_candidates(self, df, entity_col, candidates_col, codes_col):
        if any(col not in df for col in [entity_col, candidates_col, codes_col]):
            raise ValueError("Specified columns not found in the DataFrame")
        for index in tqdm(df.index, desc="Reranking candidates"):
            entity, candidates = df.at[index, entity_col], df.at[index, candidates_col]
            scores = self.predict([[entity, candidate] for candidate in candidates])
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            df.at[index, candidates_col] = [candidates[i] for i in sorted_indices]
            df.at[index, codes_col] = [df.at[index, codes_col][i] for i in sorted_indices]
        return df
