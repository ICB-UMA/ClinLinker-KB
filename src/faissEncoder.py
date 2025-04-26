import torch
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm
import numpy as np
import faiss
import pandas as pd
from huggingface_hub import login
from logger import setup_custom_logger
from typing import List, Tuple, Dict, Optional


"""
Author: Fernando Gallego, Guillermo López García & Luis Gasco Sánchez
Affiliation: Researcher at the Computational Intelligence (ICB) Group, University of Málaga & Barcelona Supercomputing Center (BSC)
"""
# Logger setup
logger = setup_custom_logger("faiss_encoder")

class FaissEncoder:
    """
    A class for encoding text using a pre-trained model and performing similarity search with FAISS indices.

    Attributes:
        model (AutoModel): Pre-trained transformer model.
        tokenizer (AutoTokenizer): Tokenizer associated with the transformer model.
        f_type (str): Type of FAISS index ("FlatL2" or "FlatIP").
        vocab (Optional[pd.DataFrame]): A DataFrame containing terms and their corresponding codes.
        max_length (int): Maximum token length for the tokenizer.
        device (str): Device to run the model on.
        faiss_index (Optional[faiss.Index]): The FAISS index for similarity search.
        verbose (int): If 1, enables progress bars with tqdm. Defaults to 0 (no progress bars).
    """

    def __init__(
        self,
        MODEL_NAME: str,
        F_TYPE: str,
        MAX_LENGTH: int,
        vocab: Optional[pd.DataFrame] = None,
        verbose: int = 0
    ):
        """
        Initializes the encoder with a specified pre-trained model, tokenizer, and FAISS index type.

        Parameters:
            MODEL_NAME (str): Name or path of the pre-trained model.
            F_TYPE (str): Type of FAISS index to use ("FlatL2" or "FlatIP").
            MAX_LENGTH (int): Maximum token length for the tokenizer.
            vocab (Optional[pd.DataFrame]): DataFrame containing terms and codes. Optional.
            verbose (int): If 1, enables progress bars with tqdm. Defaults to 0.
        """
        self.model = AutoModel.from_pretrained(MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.f_type = F_TYPE
        self.vocab = vocab
        self.max_length = MAX_LENGTH
        self.verbose = verbose
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)

        if self.vocab is not None:
            self._initialize_vocab()
        else:
            logger.warning("Vocabulary is not initialized. Ensure to set a valid vocabulary before using dependent methods.")

    def _initialize_vocab(self) -> None:
        """Initializes attributes derived from the vocabulary."""
        if not {'term', 'code'}.issubset(self.vocab.columns):
            raise ValueError("The vocabulary must contain 'term' and 'code' columns.")
        self._arr_text = self.vocab['term'].tolist()
        self._arr_codes = self.vocab['code'].tolist()
        self._arr_text_id = np.arange(len(self.vocab))
        logger.info("Vocabulary initialized successfully.")

    @property
    def arr_text(self) -> List[str]:
        if not hasattr(self, '_arr_text'):
            raise AttributeError("Vocabulary is not initialized. Use a valid vocabulary when creating the class.")
        return self._arr_text

    @property
    def arr_codes(self) -> List[str]:
        if not hasattr(self, '_arr_codes'):
            raise AttributeError("Vocabulary is not initialized. Use a valid vocabulary when creating the class.")
        return self._arr_codes

    @property
    def arr_text_id(self) -> np.ndarray:
        if not hasattr(self, '_arr_text_id'):
            raise AttributeError("Vocabulary is not initialized. Use a valid vocabulary when creating the class.")
        return self._arr_text_id

    def encode(
        self,
        texts: List[str],
        batch_size: int
    ) -> np.ndarray:
        """
        Encodes a list of texts into embeddings using the transformer model.

        Parameters:
            texts (List[str]): List of text strings to encode.
            batch_size (int): Batch size for processing.

        Returns:
            np.ndarray: Matrix of embeddings.
        """
        if not texts:
            raise ValueError("The list of texts is empty.")

        all_embeddings = []
        model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model

        progress = tqdm(range(0, len(texts), batch_size), desc="Encoding texts", disable=self.verbose == 0)
        for i in progress:
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    def fit_faiss(
        self,
        batch_size: int = 32
    ) -> None:
        """
        Fits the FAISS index using embeddings from the vocabulary.

        Parameters:
            batch_size (int): Batch size for encoding. Defaults to 32.
        """
        embeddings = self.encode(self.arr_text, batch_size).astype('float32')
        index_cls = faiss.IndexFlatL2 if self.f_type == "FlatL2" else faiss.IndexFlatIP
        index = index_cls(embeddings.shape[1])

        if self.f_type == "FlatIP":
            faiss.normalize_L2(embeddings)

        self.faiss_index = faiss.IndexIDMap(index)
        self.faiss_index.add_with_ids(embeddings, self.arr_text_id)
        logger.info("FAISS index fitted successfully.")

    def get_candidates(
        self,
        texts: List[str],
        k: int = 200,
        batch_size: int = 64
    ) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
        """
        Retrieves the top-k candidates from the FAISS index for the given texts.

        Parameters:
            texts (List[str]): Texts to search against the FAISS index.
            k (int): Number of top candidates to return. Defaults to 200.
            batch_size (int): Batch size for encoding. Defaults to 64.

        Returns:
            Tuple: Unique candidates, corresponding codes, and similarity scores.
        """
        if not hasattr(self, 'faiss_index'):
            raise AttributeError("FAISS index is not initialized. Run 'fit_faiss' first.")

        embeddings = self.encode(texts, batch_size).astype('float32')
        if self.f_type == "FlatIP":
            faiss.normalize_L2(embeddings)

        sim, indices = self.faiss_index.search(embeddings, k)
        return self._process_results(indices, sim, k)

    def _process_results(
        self,
        indices: np.ndarray,
        sim: np.ndarray,
        k: int
    ) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
        """
        Processes FAISS search results to ensure unique candidates.

        Parameters:
            indices (np.ndarray): Candidate indices from FAISS search.
            sim (np.ndarray): Similarity scores for candidates.
            k (int): Number of top unique candidates to return.

        Returns:
            Tuple: Lists of unique candidates, their codes, and similarity scores.
        """
        candidates, candidates_codes, candidates_sims = [], [], []

        for idx_list, sim_list in zip(indices, sim):
            seen = set()
            unique_candidates, unique_codes, unique_sims = [], [], []
            for idx, score in zip(idx_list, sim_list):
                if idx >= 0 and self.arr_codes[idx] not in seen:
                    seen.add(self.arr_codes[idx])
                    unique_candidates.append(self.arr_text[idx])
                    unique_codes.append(self.arr_codes[idx])
                    unique_sims.append(score)
                    if len(unique_candidates) == k:
                        break
            candidates.append(unique_candidates)
            candidates_codes.append(unique_codes)
            candidates_sims.append(unique_sims)

        return candidates, candidates_codes, candidates_sims
    

    def upload_to_hf(
        self, 
        repo_name: str, 
        token: str = None, 
        private: bool = True):
        """
        Uploads the current model and tokenizer to the Hugging Face Hub and can make the repository private.

        Parameters:
            repo_name (str): Name of the Hugging Face repository (e.g., "username/repo_name").
            token (str): Hugging Face authentication token. If not provided, will prompt for login.
            private (bool): Whether to make the repository private. Defaults to True.
        """
        if token:
            login(token=token)
        else:
            login()  # Prompt for token if not provided

        # Upload model and tokenizer to the specified Hugging Face repository with the private option
        self.model.push_to_hub(repo_name, private=private)
        self.tokenizer.push_to_hub(repo_name, private=private)
        print(f"Model and tokenizer uploaded to the repository {repo_name} on Hugging Face (private={private}).")
