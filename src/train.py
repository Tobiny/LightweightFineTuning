import os
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW, get_linear_schedule_with_warmup
from data_preparation import load_data, preprocess_data, split_data, create_text_representation
from model_utils import select_features, evaluate_model, load_bert_model, create_peft_model, get_lora_config, plot_confusion_matrix
from tqdm import tqdm
import logging
from datetime import datetime
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def setup_logging():
    log_dir = '../logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    if os.path.exists(log_file):
        print(f"Log file created: {log_file}")
    else:
        print(f"Failed to create log file: {log_file}")

    return logging.getLogger(__name__)

def check_and_create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"Created directory: {path}")
    else:
        logging.info(f"Directory already exists: {path}")

def evaluate_bert(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
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

def train_bert_model(model, train_dataloader, val_dataloader, device, class_weights, num_epochs=3):
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    best_f1 = 0
    best_model = None
    patience = 2
    no_improve = 0

    for epoch in range(num_epochs):
        logging.info(f"Training epoch {epoch + 1}/{num_epochs}...")
        model.train()
        total_loss = 0
        for batch in tqdm(train_dataloader, desc="Training"):
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}

            outputs = model(**inputs)
            logits = outputs.logits
            loss = torch.nn.functional.cross_entropy(logits, inputs['labels'], weight=class_weights.to(device))
            total_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(train_dataloader)
        logging.info(f"Average loss for epoch {epoch + 1}: {avg_loss:.4f}")

        logging.info(f"Evaluating after epoch {epoch + 1}...")
        metrics = evaluate_bert(model, val_dataloader, device)

        logging.info(f"Epoch {epoch + 1}/{num_epochs} metrics:")
        for metric, value in metrics.items():
            logging.info(f"{metric}: {value:.4f}")

        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_model = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            logging.info(f"Early stopping triggered after epoch {epoch + 1}")
            break

    return best_model

def save_model(model, path):
    try:
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(model, path)
        logging.info(f"Model saved successfully to {path}")
    except Exception as e:
        logging.error(f"Error saving model to {path}: {str(e)}")
        logging.error(f"Current working directory: {os.getcwd()}")
        logging.error(f"Directory exists: {os.path.exists(directory)}")
        logging.error(f"Directory is writable: {os.access(directory, os.W_OK)}")

def main():
    logger = setup_logging()
    logger.info("Setup initialized.")

    check_and_create_directory('../saved_models')
    check_and_create_directory('../outputs')

    logging.info("Loading and preprocessing data...")
    df = load_data('../data/heart_2022_no_nans.csv')
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    logging.info("Creating text representations...")
    X_train_text = create_text_representation(X_train)
    X_test_text = create_text_representation(X_test)

    logging.info("Loading BERT model and tokenizer...")
    tokenizer, bert_model = load_bert_model()

    logging.info("Tokenizing data...")
    train_encodings = tokenizer(X_train_text, truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(X_test_text, truncation=True, padding=True, max_length=512)

    logging.info("Creating datasets...")
    train_dataset = TensorDataset(
        torch.tensor(train_encodings['input_ids']),
        torch.tensor(train_encodings['attention_mask']),
        torch.tensor(y_train.values)
    )
    test_dataset = TensorDataset(
        torch.tensor(test_encodings['input_ids']),
        torch.tensor(test_encodings['attention_mask']),
        torch.tensor(y_test.values)
    )

    logging.info("Creating dataloaders...")
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    bert_model.to(device)

    logging.info("\nEvaluating base BERT model:")
    base_metrics = evaluate_bert(bert_model, test_dataloader, device)
    logging.info("Base BERT model performance:")
    for metric, value in base_metrics.items():
        logging.info(f"{metric}: {value:.4f}")

    logging.info("\nTraining base BERT model:")
    best_bert_model = train_bert_model(bert_model, train_dataloader, test_dataloader, device, class_weights)

    logging.info("\nCreating and training PEFT model:")
    peft_config = get_lora_config()
    peft_model = create_peft_model(bert_model, peft_config)
    peft_model.to(device)

    best_peft_model = train_bert_model(peft_model, train_dataloader, test_dataloader, device, class_weights)

    # Save the models
    bert_model_path = '../saved_models/bert_model.pth'
    peft_model_path = '../saved_models/peft_model.pth'

    logging.info("Saving models...")
    save_model(best_bert_model, bert_model_path)
    save_model(best_peft_model, peft_model_path)

    logging.info("\nTraining and evaluation complete.")

if __name__ == "__main__":
    main()