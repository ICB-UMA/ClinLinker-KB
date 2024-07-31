import os
import argparse
import pandas as pd
import pickle
import sys

# Custom module imports
sys.path.append(os.path.join(os.getcwd(), '../src'))
import faissEncoder as faiss_enc
from crossEncoder import CrossEncoderReranker
from tripletsGeneration import SimilarityHardTriplets, TopHardTriplets
from logger import setup_custom_logger
import utils

def parse_args():
    """
    Parse command line arguments.
    Returns:
        Namespace of command line arguments
    """
    parser = argparse.ArgumentParser(description="Train a cross-encoder model for entity linking.")
    parser.add_argument('--corpus', type=str, default='MedProcNER', help='Name of the corpus to process')
    parser.add_argument('--model_path', type=str, default='../../../models/NEL/spanish_sapbert_models/sapbert_15_noparents_1epoch/', help='Model path for FAISS encoder and cross encoder')
    parser.add_argument('--hard_triplets_type', type=str, choices=['top', 'sim'], default='sim', help='Type of hard triplets to generate')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length for the model')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--candidates', type=int, default=200, help='Number of candidates to generate')
    parser.add_argument('--num_negatives', type=int, default=200, help='Number of negative samples')
    parser.add_argument('--f_type', type=str, default='FlatIP', help='FAISS index type')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Optimizer weight decay')
    parser.add_argument('--eval_steps', type=int, default=250000, help='Evaluation steps')
    parser.add_argument('--test_size', default=None, type=float, help='Fraction of the data to be used as test set (optional)')
    parser.add_argument('--log_file', type=str, default=None, help='File to log to (defaults to console if not provided)')
    return parser.parse_args()



def prepare_model(args, gaz_df, train_df, logger):
    """
    Prepare and return the FAISS encoder and initial candidate generation.
    Args:
        args: Parsed command line arguments
        gaz_df: Gazetteer DataFrame
        train_df: Training link DataFrame
        logger: Configured logger object
    Returns:
        Initialized FAISS encoder and updated train_df with candidates
    """
    logger.info("Preparing model...")
    faiss_encoder = faiss_enc.FaissEncoder(args.model_path, args.f_type, args.max_length, gaz_df)
    faiss_encoder.fitFaiss()
    candidates, codes, similarities = faiss_encoder.getCandidates(train_df["term"].tolist(), args.candidates, args.max_length)
    train_df["candidates"], train_df["codes"], train_df["similarities"] = candidates, codes, similarities
    del faiss_encoder

def generate_triplets(args, train_df, logger):
    """
    Generate hard triplets based on specified method in the arguments.
    Args:
        args: Parsed command line arguments
        train_df: DataFrame with training data links
        logger: Configured logger object
    Returns:
        DataFrame containing generated hard triplets
    """
    logger.info("Generating triplets...")
    if args.hard_triplets_type == 'top':
        return TopHardTriplets(train_df).generate_triplets(args.num_negatives)
    elif args.hard_triplets_type == 'sim':
        return SimilarityHardTriplets(train_df).generate_triplets(similarity_threshold=0.35)
    else:
        logger.error("Unsupported triplets type")
        raise ValueError("Unsupported triplets type")

def train_cross_encoder(args, df_hard_triplets, logger):
    """
    Train the cross-encoder model using the generated hard triplets and save the model.
    Args:
        args: Parsed command line arguments
        df_hard_triplets: DataFrame containing the hard triplets for training
        logger: Configured logger object
    """
    logger.info("Training cross-encoder...")
    cross_encoder = CrossEncoderReranker(args.model_path, model_type="mask", max_seq_length=args.max_length)
    output_path = os.path.join("../../../models/NEL/cross-encoders/Spanish_SapBERT_noparents/", f"cef_{args.corpus.lower()}_{args.hard_triplets_type}_cand_{args.num_negatives}_epoch_{args.epochs}_bs_{args.batch_size}")
    cross_encoder.train(df_hard_triplets, output_path, args.batch_size, args.epochs, optimizer_parameters={"lr": args.lr}, weight_decay=args.weight_decay, evaluation_steps=args.eval_steps, save_best_model=False, test_size=args.test_size)
    cross_encoder.save(output_path)
    logger.info(f"Model saved to {output_path}")

def main():
    args = parse_args()
    logger = setup_custom_logger('crossEncoderTraining', log_file=args.log_file)
    logger.info("Starting the training process...")
    logger.info("Loading data...")
    _, train_df, gaz_df = utils.load_corpus_data(args.corpus)
    prepare_model(args, gaz_df, train_df, logger)
    df_hard_triplets = generate_triplets(args, train_df, logger)
    train_cross_encoder(args, df_hard_triplets, logger)
    logger.info("Training complete.")

if __name__ == "__main__":
    main()
