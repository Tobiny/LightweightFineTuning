import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW, get_linear_schedule_with_warmup
from data_preparation import prepare_data
from model_utils import load_bert_model, create_peft_model, get_lora_config, evaluate_model, plot_training_history
from tqdm import tqdm
import logging
from datetime import datetime
import numpy as np


def setup_logging():
    log_dir = '../logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

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


def train_model(model, train_dataloader, val_dataloader, device, num_epochs=3):
    try:
        logger.info("Starting model training")
        optimizer = AdamW(model.parameters(), lr=2e-5)
        total_steps = len(train_dataloader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}

        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0
            correct_train_preds = 0
            total_train_samples = 0

            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                batch = tuple(t.to(device) for t in batch)
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}

                model.zero_grad()
                outputs = model(**inputs)
                loss = outputs.loss
                logits = outputs.logits

                total_train_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct_train_preds += (preds == inputs['labels']).sum().item()
                total_train_samples += inputs['labels'].size(0)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_train_loss / len(train_dataloader)
            train_accuracy = correct_train_preds / total_train_samples
            history['train_loss'].append(avg_train_loss)
            history['train_accuracy'].append(train_accuracy)

            # Validation
            model.eval()
            total_val_loss = 0
            correct_val_preds = 0
            total_val_samples = 0

            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc="Validation"):
                    batch = tuple(t.to(device) for t in batch)
                    inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}

                    outputs = model(**inputs)
                    loss = outputs.loss
                    logits = outputs.logits

                    total_val_loss += loss.item()
                    preds = torch.argmax(logits, dim=1)
                    correct_val_preds += (preds == inputs['labels']).sum().item()
                    total_val_samples += inputs['labels'].size(0)

            avg_val_loss = total_val_loss / len(val_dataloader)
            val_accuracy = correct_val_preds / total_val_samples
            history['val_loss'].append(avg_val_loss)
            history['val_accuracy'].append(val_accuracy)

            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), '../saved_models/best_model.pth')
                logger.info("Best model saved")

        plot_training_history(history)
        logger.info("Training completed")
        return model, history

    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise


def main():
    try:
        logger.info("Starting main training process")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        # Prepare data
        file_path = '../data/heart_2022_no_nans.csv'
        X_train_text, X_val_text, y_train, y_val = prepare_data(file_path)

        # Load BERT model and tokenizer
        tokenizer, base_model = load_bert_model()

        # Tokenize data
        train_encodings = tokenizer(X_train_text, truncation=True, padding=True, max_length=512)
        val_encodings = tokenizer(X_val_text, truncation=True, padding=True, max_length=512)

        # Create datasets
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

        # Create dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=16)

        # Train base model
        base_model.to(device)
        logger.info("Training base BERT model")
        trained_base_model, base_history = train_model(base_model, train_dataloader, val_dataloader, device)

        # Create and train PEFT model
        peft_config = get_lora_config()
        peft_model = create_peft_model(base_model, peft_config)
        peft_model.to(device)
        logger.info("Training PEFT model")
        trained_peft_model, peft_history = train_model(peft_model, train_dataloader, val_dataloader, device)

        # Save models
        torch.save(trained_base_model.state_dict(), '../saved_models/final_base_model.pth')
        torch.save(trained_peft_model.state_dict(), '../saved_models/final_peft_model.pth')
        logger.info("Final models saved")

        logger.info("Training process completed successfully")

    except Exception as e:
        logger.error(f"Error in main training process: {str(e)}")


if __name__ == "__main__":
    main()