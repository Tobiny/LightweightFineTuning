import logging
from datetime import datetime
import os
from data_preparation import prepare_data
from train import train_model
from evaluate import evaluate_model_performance, compare_models
from model_utils import load_bert_model, create_peft_model, get_lora_config
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer


def setup_logging():
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'main_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()


def main():
    try:
        logger.info("Starting the main process")

        # Data preparation
        logger.info("Preparing data")
        file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'heart_2022_no_nans.csv')
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(file_path)

        # Model preparation
        logger.info("Preparing models")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer, base_model = load_bert_model()
        base_model.to(device)

        peft_config = get_lora_config()
        peft_model = create_peft_model(base_model, peft_config)
        peft_model.to(device)

        # Data tokenization and DataLoader creation
        logger.info("Tokenizing data and creating DataLoaders")
        train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=512)
        val_encodings = tokenizer(X_val, truncation=True, padding=True, max_length=512)
        test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=512)

        train_dataset = TensorDataset(
            torch.tensor(train_encodings['input_ids']),
            torch.tensor(train_encodings['attention_mask']),
            torch.tensor(y_train.values)
        )
        val_dataset = TensorDataset(
            torch.tensor(val_encodings['input_ids']),
            torch.tensor(val_encodings['attention_mask']),
            torch.tensor(y_val.values)
        )
        test_dataset = TensorDataset(
            torch.tensor(test_encodings['input_ids']),
            torch.tensor(test_encodings['attention_mask']),
            torch.tensor(y_test.values)
        )

        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=16)
        test_dataloader = DataLoader(test_dataset, batch_size=16)

        # Model training
        logger.info("Training base BERT model")
        trained_base_model, base_history = train_model(base_model, train_dataloader, val_dataloader, device)

        logger.info("Training PEFT model")
        trained_peft_model, peft_history = train_model(peft_model, train_dataloader, val_dataloader, device)

        # Model evaluation
        logger.info("Evaluating base BERT model")
        base_metrics = evaluate_model_performance(trained_base_model, test_dataloader, device)

        logger.info("Evaluating PEFT model")
        peft_metrics = evaluate_model_performance(trained_peft_model, test_dataloader, device)

        # Model comparison
        logger.info("Comparing models")
        compare_models(base_metrics, peft_metrics)

        logger.info("Main process completed successfully")

    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")


if __name__ == "__main__":
    main()