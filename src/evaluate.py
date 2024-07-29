import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from peft import PeftModel, PeftConfig
from data_preparation import load_data, preprocess_data, split_data, create_text_representation
from model_utils import evaluate_model, plot_confusion_matrix
import logging
from datetime import datetime

def setup_logging():
    log_dir = '../logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'evaluation_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger(__name__)

def load_model(model_path, device):
    try:
        model = torch.load(model_path, map_location=device)
        logging.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model from {model_path}: {str(e)}")
        return None

def evaluate_model_performance(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
            labels = batch[2]

            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics, cm = evaluate_model(all_labels, all_preds)
    plot_confusion_matrix(cm, ['No Heart Attack', 'Heart Attack'])
    return metrics

def main():
    logger = setup_logging()
    logger.info("Evaluation started")

    logger.info("Loading and preprocessing data...")
    df = load_data('../data/heart_2022_no_nans.csv')
    X, y = preprocess_data(df)
    _, X_test, _, y_test = split_data(X, y)

    logger.info("Creating text representations...")
    X_test_text = create_text_representation(X_test)

    logger.info("Loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    logger.info("Tokenizing data...")
    test_encodings = tokenizer(X_test_text, truncation=True, padding=True, max_length=512)

    logger.info("Creating dataset...")
    test_dataset = TensorDataset(
        torch.tensor(test_encodings['input_ids']),
        torch.tensor(test_encodings['attention_mask']),
        torch.tensor(y_test.values)
    )

    logger.info("Creating dataloader...")
    test_dataloader = DataLoader(test_dataset, batch_size=16)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load and evaluate base BERT model
    base_model_path = '../saved_models/bert_model.pth'
    base_model = load_model(base_model_path, device)
    if base_model:
        logger.info("\nEvaluating base BERT model:")
        base_metrics = evaluate_model_performance(base_model, test_dataloader, device)
        logger.info("Base BERT model performance:")
        for metric, value in base_metrics.items():
            logger.info(f"{metric}: {value:.4f}")

    # Load and evaluate PEFT model
    peft_model_path = '../saved_models/peft_model.pth'
    peft_model = load_model(peft_model_path, device)
    if peft_model:
        logger.info("\nEvaluating PEFT model:")
        peft_metrics = evaluate_model_performance(peft_model, test_dataloader, device)
        logger.info("PEFT model performance:")
        for metric, value in peft_metrics.items():
            logger.info(f"{metric}: {value:.4f}")

    # Compare performances
    if base_model and peft_model:
        logger.info("\nPerformance Comparison:")
        for metric in base_metrics.keys():
            base_value = base_metrics[metric]
            peft_value = peft_metrics[metric]
            difference = peft_value - base_value
            logger.info(f"{metric}: Base = {base_value:.4f}, PEFT = {peft_value:.4f}, Difference = {difference:.4f}")

    logger.info("Evaluation complete.")

if __name__ == "__main__":
    main()