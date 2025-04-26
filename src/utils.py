from tqdm.auto import tqdm
import pandas as pd
from typing import Dict, List, Tuple, Optional
from crossEncoder import CrossEncoderReranker
from metrics import calculate_topk_accuracy
from logger import setup_custom_logger
"""
Author: Fernando Gallego
Affiliation: Researcher at the Computational Intelligence (ICB) Group, University of Málaga
"""

def extract_column_names_from_ctl_file(
    ctl_file_path: str) -> List[str]:
    """
    Extract column names from a CTL file.

    Args:
        ctl_file_path (str): Path to the CTL file.

    Returns:
        List[str]: List of column names extracted from the file.
    """
    with open(ctl_file_path, 'r') as file:
        lines = file.readlines()

    # Find the index of the line containing 'trailing nullcols'
    start_index = next(i for i, line in enumerate(lines) if 'trailing nullcols' in line) + 1
    # The second last line of the file is the boundary for processing
    end_index = len(lines) - 1

    column_names = []
    for line in lines[start_index:end_index]:
        # Extract only the column name, assuming it is the first word
        name = line.split()[0].replace('(', '').replace(')', '').strip()
        if name:
            column_names.append(name)

    return column_names


def read_rrf_file_in_chunks(
    file_path: str, 
    chunk_size: int, 
    columns: List[str], 
    dtype_dict: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Read an RRF file in chunks and concatenate the chunks into a DataFrame.

    Args:
        file_path (str): Path to the RRF file.
        chunk_size (int): Number of lines per chunk.
        columns (List[str]): List of column names.
        dtype_dict (Optional[Dict[str, str]]): Dictionary specifying data types for columns.

    Returns:
        pd.DataFrame: Concatenated DataFrame containing all chunks.
    """
    chunk_list = []

    with tqdm(desc="Processing", unit="line") as pbar:
        for chunk in pd.read_csv(
            file_path,
            sep='|',
            chunksize=chunk_size,
            na_filter=False,
            low_memory=True,  
            dtype=dtype_dict,
            usecols=range(len(columns)),  
            names=columns,  
        ):
            chunk_list.append(chunk)
            pbar.update(len(chunk))

    df = pd.concat(chunk_list, ignore_index=True)
    return df

def load_corpus_data(
    base_path: str, 
    corpus: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load test, train, and gazetteer data for a specific corpus.

    Args:
        base_path (str): Base path containing the corpus directories.
        corpus (str): Name of the corpus ("SympTEMIST", "MedProcNER", or "DisTEMIST").

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Test, train, and gazetteer DataFrames.
    
    Raises:
        ValueError: If the corpus is not supported.
    """
    corpus_paths = {
        "SympTEMIST": {
            "test": f"{base_path}/SympTEMIST/symptemist-complete_240208/symptemist_test/subtask2-linking/symptemist_tsv_test_subtask2.tsv",
            "train": f"{base_path}/SympTEMIST/symptemist-complete_240208/symptemist_train/subtask2-linking/symptemist_tsv_train_subtask2_complete.tsv",
            "gaz": f"{base_path}/SympTEMIST/symptemist-complete_240208/symptemist_gazetteer/symptemist_gazetter_snomed_ES_v2.tsv"
        },
        "MedProcNER": {
            "test": f"{base_path}/MedProcNER/medprocner_gs_train+test+gazz+multilingual+crossmap_230808/medprocner_test/tsv/medprocner_tsv_test_subtask2.tsv",
            "train": f"{base_path}/MedProcNER/medprocner_gs_train+test+gazz+multilingual+crossmap_230808/medprocner_train/tsv/medprocner_tsv_train_subtask2.tsv",
            "gaz": f"{base_path}/MedProcNER/medprocner_gs_train+test+gazz+multilingual+crossmap_230808/medprocner_gazetteer/gazzeteer_medprocner_v1_noambiguity.tsv"
        },
        "DisTEMIST": {
            "test": f"{base_path}/DisTEMIST/distemist_zenodo/test_annotated/subtrack2_linking/distemist_subtrack2_test_linking.tsv",
            "train": f"{base_path}/DisTEMIST/distemist_zenodo/training/subtrack2_linking/distemist_subtrack2_training2_linking.tsv",
            "gaz": f"{base_path}/DisTEMIST/dictionary_distemist.tsv"
        }
    }

    if corpus not in corpus_paths:
        raise ValueError(f"Unsupported corpus: {corpus}")

    paths = corpus_paths[corpus]
    test_df = pd.read_csv(paths["test"], sep="\t", dtype={"code": str})
    train_df = pd.read_csv(paths["train"], sep="\t", dtype={"code": str})
    df_gaz = pd.read_csv(paths["gaz"], sep="\t", dtype={"code": str})

    if corpus == "SympTEMIST" or corpus == "MedProcNER":
        test_df.rename(columns={'text': 'term'}, inplace=True)
        train_df.rename(columns={'text': 'term'}, inplace=True)
    elif corpus == "DisTEMIST":
        test_df.rename(columns={'span': 'term'}, inplace=True)
        train_df.rename(columns={'span': 'term'}, inplace=True)

    return test_df, train_df, df_gaz


def evaluate_model(
    model_name: str,
    f_type: str,
    max_length: int,
    train_gaz_df,
    gs_df,
    clean_df,
    um_df,
    uc_df,
    gs_results: dict,
    clean_results: dict,
    um_results: dict,
    uc_results: dict,
    model_map: dict,
    top_k_values: list,
    corpus: str
) -> tuple:
    """
    Evaluates a given model (bi-encoder) and, if applicable, its associated cross-encoder.
    Saves the top-k results in the corresponding result dictionaries.
    """
    # Initialize encoder and fit FAISS
    faiss_encoder = faiss_enc.FaissEncoder(model_name, f_type, max_length, train_gaz_df)
    faiss_encoder.fit_faiss()

    # Retrieve candidates using FAISS
    gs_preds = gs_df.copy()
    gs_preds["candidates"], gs_preds["codes"], _ = faiss_encoder.get_candidates(gs_df["term"].tolist(), k=200)

    clean_preds = clean_df.copy()
    clean_preds["candidates"], clean_preds["codes"], _ = faiss_encoder.get_candidates(clean_df["term"].tolist(), k=200)

    um_preds = um_df.copy()
    um_preds["candidates"], um_preds["codes"], _ = faiss_encoder.get_candidates(um_df["term"].tolist(), k=200)

    uc_preds = uc_df.copy()
    uc_preds["candidates"], uc_preds["codes"], _ = faiss_encoder.get_candidates(uc_df["term"].tolist(), k=200)

    # Define the model key for the results
    model_key = model_map.get(model_name.split("/")[-1], "None")

    # Save FAISS results
    gs_results[model_key] = calculate_topk_accuracy(gs_preds, top_k_values)
    clean_results[model_key] = calculate_topk_accuracy(clean_preds, top_k_values)
    um_results[model_key] = calculate_topk_accuracy(um_preds, top_k_values)
    uc_results[model_key] = calculate_topk_accuracy(uc_preds, top_k_values)

    # Check if re-ranking with a cross-encoder is required
    if model_key in ["ClinLinker", "ClinLinker-KB-P", "ClinLinker-KB-GP"]:
        # Define the cross-encoder model path template
        template_map = {
            "ClinLinker": "Spanish_SapBERT_noparents",
            "ClinLinker-KB-P": "Spanish_SapBERT_parents",
            "ClinLinker-KB-GP": "Spanish_SapBERT_grandparents"
        }

        template = template_map[model_key]

        ce_path = f"/scratch/models/NEL/cross-encoders/{template}/cef_{corpus.lower()}_{template}_sim_cand_200_epoch_1_bs_128"

        ce_reranker = CrossEncoderReranker(model_name=ce_path, model_type="st", max_seq_length=max_length)

        gs_results[model_key + "_CE"] = calculate_topk_accuracy(
            ce_reranker.rerank_candidates(gs_preds.copy(deep=True), "term", "candidates", "codes"),
            top_k_values
        )
        clean_results[model_key + "_CE"] = calculate_topk_accuracy(
            ce_reranker.rerank_candidates(clean_preds.copy(deep=True), "term", "candidates", "codes"),
            top_k_values
        )
        um_results[model_key + "_CE"] = calculate_topk_accuracy(
            ce_reranker.rerank_candidates(um_preds.copy(deep=True), "term", "candidates", "codes"),
            top_k_values
        )
        uc_results[model_key + "_CE"] = calculate_topk_accuracy(
            ce_reranker.rerank_candidates(uc_preds.copy(deep=True), "term", "candidates", "codes"),
            top_k_values
        )

        return gs_results, clean_results, um_results, uc_results




def calculate_norm(df_gs, df_preds, log_file=None):
    #Function adapted from https://github.com/TeMU-BSC/medprocner_evaluation_library/blob/main/utils.py

    logger = setup_custom_logger('norm_logger', log_file) if log_file else None
    if logger:
        logger.info("Computing evaluation scores for Task 2 (norm)")    
        logger.info(f"Number of NO_CODE: { df_gs[df_gs['code'] == 'NO_CODE'].shape[0]}")
        composite_count = df_gs[df_gs['code'].str.contains(r'\+')].shape[0]
        logger.info(f"Number of COMPOSITE: {composite_count}")
    list_gs_per_doc = df_gs.groupby('filename', group_keys=False).apply(lambda x: x[['filename', 'span_ini', 'span_end', "term", "label", "code"]].values.tolist()).to_list()
    list_preds_per_doc = df_preds.groupby('filename', group_keys=False).apply(lambda x: x[['filename', 'span_ini', 'span_end', "term", "label", "code"]].values.tolist()).to_list()
    scores = calculate_fscore(list_gs_per_doc, list_preds_per_doc, 'norm', logger)
    return scores

def calculate_ner(df_gs, df_preds, log_file=None):
    #Function adapted from https://github.com/TeMU-BSC/medprocner_evaluation_library/blob/main/utils.py
    logger = setup_custom_logger('ner_logger', log_file) if log_file else None
    if logger:
        logger.info("Computing evaluation scores for Task 1 (ner)")
        logger.info(f"Number of NO_CODE: { df_gs[df_gs['code'] == 'NO_CODE'].shape[0]}")
        composite_count = df_gs[df_gs['code'].str.contains(r'\+')].shape[0]
        logger.info(f"Number of COMPOSITE: {composite_count}")
    list_gs_per_doc = df_gs.groupby('filename').apply(lambda x: x[["filename", 'span_ini', 'span_end', "term",  "label"]].values.tolist()).to_list()
    list_preds_per_doc = df_preds.groupby('filename').apply(lambda x: x[["filename", 'span_ini', 'span_end', "term", "label"]].values.tolist()).to_list()
    scores = calculate_fscore(list_gs_per_doc, list_preds_per_doc, 'ner', logger)
    return scores



def calculate_fscore(gold_standard, predictions, task, logger=None):
    # Code from https://github.com/TeMU-BSC/medprocner_evaluation_library/blob/main/utils.py
    """
    Calculate micro-averaged precision, recall and f-score from two pandas dataframe
    Depending on the task, do some different pre-processing to the data
    """
    if logger:
        logger.info("Starting F-score calculation")


    # Cumulative true positives, false positives, false negatives
    total_tp, total_fp, total_fn = 0, 0, 0
    # Dictionary to store files in gold and prediction data.
    gs_files = {}
    pred_files = {}
    for document in gold_standard:
        document_id = document[0][0]
        gs_files[document_id] = document
    for document in predictions:
        document_id = document[0][0]
        pred_files[document_id] = document

    # Dictionary to store scores
    scores = {}

    # Iterate through documents in the Gold Standard
    for document_id in gs_files.keys():
        doc_tp, doc_fp, doc_fn = 0, 0, 0
        gold_doc = gs_files[document_id]
        #  Check if there are predictions for the current document, default to empty document if false
        if document_id not in pred_files.keys():
            predicted_doc = []
        else:
            predicted_doc = pred_files[document_id]
        if task == 'index':  # Separate codes
            gold_doc = list(set(gold_doc[0][1].split('+')))
            predicted_doc = list(set(predicted_doc[0][1].split('+'))) if predicted_doc else []
        # Iterate through a copy of our gold mentions
        for gold_annotation in gold_doc[:]:
            # Iterate through predictions looking for a match
            for prediction in predicted_doc[:]:
                # Separate possible composite normalizations
                if task == 'norm':
                    separate_prediction = prediction[:-1] + [code.rstrip() for code in sorted(str(prediction[-1]).split('+'))]  # Need to sort
                    separate_gold_annotation = gold_annotation[:-1] + [code.rstrip() for code in str(gold_annotation[-1]).split('+')]
                    if logger:
                        logger.info(f"Gold annotation: {set(separate_gold_annotation)}, Prediction: {set(separate_prediction)}")

                    if set(separate_gold_annotation) == set(separate_prediction):
                        # Add a true positive
                        doc_tp += 1
                        # Remove elements from list to calculate later false positives and false negatives
                        predicted_doc.remove(prediction)
                        gold_doc.remove(gold_annotation)
                        break
                if logger:
                    logger.info(f"Gold annotation: {set(gold_annotation)}, Prediction: {set(prediction)}")

                if set(gold_annotation) == set(prediction):
                    # Add a true positive
                    doc_tp += 1
                    # Remove elements from list to calculate later false positives and false negatives
                    predicted_doc.remove(prediction)
                    gold_doc.remove(gold_annotation)
                    break
        # Get the number of false positives and false negatives from the items remaining in our lists
        doc_fp += len(predicted_doc)
        doc_fn += len(gold_doc)
        if logger:
            logger.info(f"Document ID: {document_id}, TP: {doc_tp}, FP: {doc_fp}, FN: {doc_fn}")

        # Calculate document score
        try:
            precision = doc_tp / (doc_tp + doc_fp)
        except ZeroDivisionError:
            precision = 0
        try:
            recall = doc_tp / (doc_tp + doc_fn)
        except ZeroDivisionError:
            recall = 0
        if precision == 0 or recall == 0:
            f_score = 0
        else:
            f_score = 2 * precision * recall / (precision + recall)
        # Add to dictionary
        scores[document_id] = {"recall": round(recall, 4), "precision": round(precision, 4), "f_score": round(f_score, 4)}
        if logger:
            logger.info(f"Scores for document {document_id}: {scores[document_id]}")
        # Update totals
        total_tp += doc_tp
        total_fn += doc_fn
        total_fp += doc_fp

    # Now let's calculate the micro-averaged score using the cumulative TP, FP, FN
    try:
        precision = total_tp / (total_tp + total_fp)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = total_tp / (total_tp + total_fn)
    except ZeroDivisionError:
        recall = 0
    if precision == 0 or recall == 0:
        f_score = 0
    else:
        f_score = 2 * precision * recall / (precision + recall)

    scores['total'] = {"recall": round(recall, 4), "precision": round(precision, 4), "f_score": round(f_score, 4)}
    if logger:
        logger.info(f"Final scores: {scores['total']}")

    return scores