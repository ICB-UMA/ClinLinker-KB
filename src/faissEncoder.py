import torch
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm
import numpy as np
import faiss
import pandas as pd


"""
Author: Luis Gasco Sánchez, Guillermo López García & Fernando Gallego
Affiliation: Researcher at the Barcelona Supercomputing Center (BSC), Department of Computational Biomedicine, Cedars-Sinai Medical Center & Computational Intelligence (ICB) Group, University of Málaga
"""

class FaissEncoder:
    """
    A class to encode text using a specified transformer model and fit/search using FAISS indices.

    Attributes:
        model (AutoModel): The pre-trained transformer model.
        tokenizer (AutoTokenizer): Tokenizer for the transformer model.
        f_type (str): Type of FAISS index to use. Options include "FlatL2" and "FlatIP".
        vocab (DataFrame): A pandas DataFrame containing terms and their corresponding codes.
        arr_text (list): List of terms from vocab.
        arr_codes (list): List of codes corresponding to terms in arr_text.
        arr_text_id (ndarray): Array of indices for arr_text.
        device (str): Device to run the model on.
        faiss_index (Index): The FAISS index for searching encoded texts.
    """
    def __init__(self, MODEL_NAME: str, F_TYPE: str, MAX_LENGTH: int, vocab: pd.DataFrame):
        """
        Initializes the encoder with a model and tokenizer, sets up device and prepares the FAISS index type.

        Parameters:
            MODEL_NAME (str): The name or path of the pre-trained model.
            F_TYPE (str): The type of FAISS index to use.
            MAX_LENGTH (int): Maximum length of tokens for the tokenizer.
            vocab (DataFrame): DataFrame containing terms and corresponding codes.
        """
        self.model = AutoModel.from_pretrained(MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.f_type = F_TYPE
        self.vocab = vocab
        self.arr_text = self.vocab['term'].values.tolist()
        self.arr_codes = self.vocab['code'].values.tolist()
        self.arr_text_id = np.arange(len(self.vocab))
        self.max_length = MAX_LENGTH

        # Setup device and DataParallel
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)

    def encode(self, texts, batch_size):
        """
        Encodes the given texts into embeddings using the transformer model.

        Parameters:
            texts (list): List of text strings to encode.
            batch_size (int): The size of each batch for processing.

        Returns:
            ndarray: A numpy array of embeddings.
        """
        all_embeddings = []
        num_batches = (len(texts) + batch_size - 1) // batch_size
        model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model

        for batch_idx in tqdm(range(num_batches), desc="Encoding"):
            batch_texts = texts[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()
                all_embeddings.append(embeddings)
        return np.vstack(all_embeddings)

    def fitFaiss(self, batch_size=32):
        """
        Fits the FAISS index using encoded embeddings of the vocabulary terms.

        Parameters:
            batch_size (int): The size of each batch for encoding texts. Defaults to 32.

        Effects:
            Initializes and stores the FAISS index with embeddings and associated IDs.
        """
        embeddings = self.encode(self.arr_text, batch_size)
        embeddings = embeddings.astype('float32')
        index_type = faiss.IndexFlatL2 if self.f_type == "FlatL2" else faiss.IndexFlatIP
        faiss_index = faiss.IndexIDMap(index_type(embeddings.shape[1]))

        if self.f_type == "FlatIP":
            faiss.normalize_L2(embeddings)
        
        faiss_index.add_with_ids(embeddings, self.arr_text_id)
        self.faiss_index = faiss_index

    def getCandidates(self, texts, k=200, batch_size=64):
        """
        Searches the FAISS index to find top candidate terms for the given texts.

        Parameters:
            texts (list): List of texts to search against the FAISS index.
            k (int): Number of top candidates to retrieve for each text.
            batch_size (int): Batch size to use for encoding texts.

        Returns:
            tuple: Returns three lists containing the candidates, their codes, and similarity scores.
        """
        encoded_entities = self.encode(texts, batch_size).astype('float32')
        if self.f_type == "FlatIP":
            faiss.normalize_L2(encoded_entities)

        sim, indexes = self.faiss_index.search(encoded_entities, k * 3)
        return self._process_results(indexes, sim, k)

    def _process_results(self, indexes, sim, k):
        """
        Processes the raw results from FAISS search to ensure uniqueness and order by similarity.

        Parameters:
            indexes (ndarray): Indices of the candidates from the FAISS search.
            sim (ndarray): Similarity scores corresponding to the indexes.
            k (int): Number of top unique candidates to return.

        Returns:
            tuple: Three lists containing the unique candidates, their codes, and similarity scores.
        """
        candidates, candidates_codes, candidates_sims = [], [], []
        for idx_list, sim_list in zip(indexes, sim):
            seen_codes, unique_candidates, unique_codes, unique_sims = set(), [], [], []
            for idx, sim_score in zip(idx_list, sim_list):
                if idx < len(self.arr_text) and self.arr_codes[idx] not in seen_codes:
                    seen_codes.add(self.arr_codes[idx])
                    unique_candidates.append(self.arr_text[idx])
                    unique_codes.append(self.arr_codes[idx])
                    unique_sims.append(sim_score)
                    if len(unique_candidates) == k:
                        break
            candidates.append(unique_candidates)
            candidates_codes.append(unique_codes)
            candidates_sims.append(unique_sims)
        return candidates, candidates_codes, candidates_sims

    def evaluate(self, eval_df, k_values, batch_size=64):
        """
        Evaluates the precision of the FAISS index for given terms at multiple k-values.

        Parameters:
            eval_df (DataFrame): DataFrame containing the terms and correct codes for evaluation.
            k_values (list): List of k-values at which to evaluate precision.
            batch_size (int): Batch size to use for encoding texts.

        Returns:
            dict: A dictionary mapping each k-value to its corresponding precision.
        """
        _, candidates_codes, _ = self.getCandidates(eval_df['term'].tolist(), max(k_values) + 150, batch_size)
        precision_at_k = {}
        for k in k_values:
            correct_predictions = sum(1 for true_code, candidates in zip(eval_df['code'], candidates_codes) if true_code in candidates[:k])
            precision_at_k[k] = correct_predictions / len(eval_df)
        return precision_at_k