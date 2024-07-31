from tqdm.auto import tqdm
import pandas as pd
from typing import List, Dict
from crossEncoder import CrossEncoderReranker
from metrics import calculate_topk_accuracy
from logger import setup_custom_logger
"""
Author: Fernando Gallego
Affiliation: Researcher at the Computational Intelligence (ICB) Group, University of Málaga
"""

def extract_column_names_from_ctl_file(ctl_file_path):
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


def read_rrf_file_in_chunks(file_path, chunk_size, columns, dtype_dict=None):
    total_lines = sum(1 for line in open(file_path, 'r', encoding='utf8'))
    chunk_list = []

    with tqdm(total=total_lines, desc="Processing", unit="line") as pbar:
        for chunk in pd.read_csv(file_path, sep='|', chunksize=chunk_size, na_filter=False, low_memory=False, dtype=dtype_dict):
            chunk = chunk.iloc[:, :len(columns)]
            chunk_list.append(chunk)
            pbar.update(min(chunk_size, total_lines - pbar.n))

    df = pd.concat(chunk_list, axis=0)
    df.columns = columns
    return df

def load_corpus_data(corpus):
    """Load testing data and gazetteer based on the specified corpus."""
    if corpus == "SympTEMIST":
        test_df = pd.read_csv("../../../data/SympTEMIST/symptemist-complete_240208/symptemist_test/subtask2-linking/symptemist_tsv_test_subtask2.tsv", sep="\t", header=0, dtype={"code": str})
        test_df = test_df.rename(columns={'text': 'term'})
        df_gaz = pd.read_csv("../../../data/SympTEMIST/symptemist-complete_240208/symptemist_gazetteer/symptemist_gazetter_snomed_ES_v2.tsv", sep="\t", header=0, dtype={"code": str})
        train_df = pd.read_csv("../../../data/SympTEMIST/symptemist-complete_240208/symptemist_train/subtask2-linking/symptemist_tsv_train_subtask2_complete.tsv", sep="\t", header=0, dtype={"code": str})
        train_df = train_df.rename(columns={'text': 'term'})
    elif corpus == "MedProcNER":
        test_df = pd.read_csv("../../../data/MedProcNER/medprocner_gs_train+test+gazz+multilingual+crossmap_230808/medprocner_test/tsv/medprocner_tsv_test_subtask2.tsv", sep="\t", header=0, dtype={"code": str})
        test_df = test_df.rename(columns={'text': 'term'})
        df_gaz = pd.read_csv("../../../data/MedProcNER/medprocner_gs_train+test+gazz+multilingual+crossmap_230808/medprocner_gazetteer/gazzeteer_medprocner_v1_noambiguity.tsv", sep="\t", header=0, dtype={"code": str})
        train_df = pd.read_csv("../../../data/MedProcNER/medprocner_gs_train+test+gazz+multilingual+crossmap_230808/medprocner_train/tsv/medprocner_tsv_train_subtask2.tsv", sep="\t", header=0, dtype={"code": str})
        train_df = train_df.rename(columns={'text': 'term'})
    elif corpus == "DisTEMIST":
        test_df = pd.read_csv("../../../data/DisTEMIST/distemist_zenodo/test_annotated/subtrack2_linking/distemist_subtrack2_test_linking.tsv", sep="\t", header=0, dtype={"code": str})
        test_df = test_df.rename(columns={'span': 'term'})
        df_gaz = pd.read_csv("../../../data/DisTEMIST/dictionary_distemist.tsv", sep="\t", header=0, dtype={"code": str})
        train_df = pd.read_csv("../../../data/DisTEMIST/distemist_zenodo/training/subtrack2_linking/distemist_subtrack2_training2_linking.tsv", sep="\t", header=0, dtype={"code": str})
        train_df = train_df.rename(columns={'span': 'term'})
    else:
        raise ValueError(f"Unsupported corpus: {corpus}")
    
    return test_df, train_df, df_gaz


def evaluate_model(MODEL_NAME: str, gs_df: pd.DataFrame, train_df: pd.DataFrame, gaz_df: pd.DataFrame, top_k_values: List[int]) -> Dict[str, Dict[int, float]]:
    """
    Evaluate the model using the provided dataframes and top_k values.
    
    :param MODEL_NAME: Name of the model to be used as a key in the result dictionaries.
    :param gs_df: Gold standard dataframe.
    :param train_df: Training dataframe.
    :param gaz_df: Gazetteer dataframe.
    :param top_k_values: List of k values to calculate the Top-k accuracy.
    :return: Three dictionaries containing the Top-k accuracy results for gs_df, uc_df, and um_df respectively.
    """
    # Create the dataframes uc_df and um_df
    uc_df = gs_df[~gs_df['code'].isin(train_df['code'])]
    um_df = gs_df[~gs_df['term'].isin(train_df['term'])]
    
    # Evaluate the dataframes gs_df, uc_df, and um_df
    gs_results = calculate_topk_accuracy(gs_df, top_k_values)
    uc_results = calculate_topk_accuracy(uc_df, top_k_values)
    um_results = calculate_topk_accuracy(um_df, top_k_values)
    
    # Create result dictionaries with the model name as key
    gs_dict = {MODEL_NAME: gs_results}
    uc_dict = {MODEL_NAME: uc_results}
    um_dict = {MODEL_NAME: um_results}
    
    return gs_dict, uc_dict, um_dict

def evaluate_crossencoder(MODEL_NAME: str, crossreranker: CrossEncoderReranker, gs_df: pd.DataFrame, train_df: pd.DataFrame, gaz_df: pd.DataFrame, top_k_values: List[int]) -> Dict[str, Dict[int, float]]:
    """
    Evaluate the model using the provided dataframes and top_k values.
    
    :param MODEL_NAME: Name of the model to be used as a key in the result dictionaries.
    :param crossreranker: Cross Encoder model.
    :param gs_df: Gold standard dataframe.
    :param train_df: Training dataframe.
    :param gaz_df: Gazetteer dataframe.
    :param top_k_values: List of k values to calculate the Top-k accuracy.
    :return: Three dictionaries containing the Top-k accuracy results for gs_df, uc_df, and um_df respectively.
    """
    # Create the dataframes uc_df and um_df
    gs_reranked_df = crossreranker.rerank_candidates(gs_df, "term", "candidates", "codes")
    uc_df = gs_df[~gs_reranked_df['code'].isin(train_df['code'])]
    um_df = gs_df[~gs_reranked_df['term'].isin(train_df['term'])]
    
    # Evaluate the dataframes gs_df, uc_df, and um_df
    gs_results = calculate_topk_accuracy(gs_reranked_df, top_k_values)
    uc_results = calculate_topk_accuracy(uc_df, top_k_values)
    um_results = calculate_topk_accuracy(um_df, top_k_values)
    
    # Create result dictionaries with the model name as key
    gs_dict = {MODEL_NAME: gs_results}
    uc_dict = {MODEL_NAME: uc_results}
    um_dict = {MODEL_NAME: um_results}
    
    return gs_dict, uc_dict, um_dict



def results2tsv(results: List[Dict[str, Dict[int, float]]]) -> pd.DataFrame:
    """
    Create a DataFrame with top-k accuracies for different models.
    
    :param results: List of dictionaries containing model names and their top-k accuracies.
                    Each dictionary should be in the format:
                    [{'ModelName': {1: accuracy1, 5: accuracy5, ..., k: accuracyk}}, ...]
    :return: DataFrame with models as rows and top-k values as columns, containing accuracies as cell values.
    """
    # Initialize a list to collect rows for the DataFrame
    rows = []

    # Process the list of dictionaries
    for result in results:
        for model_name, accuracies in result.items():
            row = {'MODEL_NAME': model_name}
            for k, accuracy in accuracies.items():
                row[k] = round(accuracy, 3)
            rows.append(row)

    # Convert the list of rows to a DataFrame
    result_df = pd.DataFrame(rows)

    return result_df


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