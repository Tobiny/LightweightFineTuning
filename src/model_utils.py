from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from transformers import BertTokenizer, BertForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from datetime import datetime


def setup_logging():
    log_dir = '../logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'model_utils_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

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


def select_features(X, y, k=20):
    try:
        logger.info(f"Selecting top {k} features")
        selector = SelectKBest(f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        selected_feature_indices = selector.get_support(indices=True)
        logger.info("Feature selection completed")
        return X_selected, selected_feature_indices
    except Exception as e:
        logger.error(f"Error in feature selection: {str(e)}")
        raise


def evaluate_model(y_true, y_pred, y_pred_proba=None):
    try:
        logger.info("Evaluating model performance")
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='weighted'),
            "recall": recall_score(y_true, y_pred, average='weighted'),
            "f1_score": f1_score(y_true, y_pred, average='weighted'),
        }
        if y_pred_proba is not None:
            metrics["auc_roc"] = roc_auc_score(y_true, y_pred_proba, average='weighted', multi_class='ovr')

        cm = confusion_matrix(y_true, y_pred)
        logger.info("Model evaluation completed")
        return metrics, cm
    except Exception as e:
        logger.error(f"Error in model evaluation: {str(e)}")
        raise


def plot_confusion_matrix(cm, class_names):
    try:
        logger.info("Plotting confusion matrix")
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('../outputs/confusion_matrix.png')
        plt.close()
        logger.info("Confusion matrix plotted and saved")
    except Exception as e:
        logger.error(f"Error in plotting confusion matrix: {str(e)}")
        raise


def load_bert_model(num_labels=2):
    try:
        logger.info("Loading BERT model and tokenizer")
        model_name = "bert-base-uncased"
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        logger.info("BERT model and tokenizer loaded successfully")
        return tokenizer, model
    except Exception as e:
        logger.error(f"Error in loading BERT model: {str(e)}")
        raise


def create_peft_model(model, peft_config):
    try:
        logger.info("Creating PEFT model")
        peft_model = get_peft_model(model, peft_config)
        logger.info("PEFT model created successfully")
        return peft_model
    except Exception as e:
        logger.error(f"Error in creating PEFT model: {str(e)}")
        raise


def get_lora_config():
    try:
        logger.info("Creating LoRA configuration")
        config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
        )
        logger.info("LoRA configuration created successfully")
        return config
    except Exception as e:
        logger.error(f"Error in creating LoRA configuration: {str(e)}")
        raise


def plot_training_history(history):
    try:
        logger.info("Plotting training history")
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history['train_accuracy'], label='Train Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig('../outputs/training_history.png')
        plt.close()
        logger.info("Training history plotted and saved")
    except Exception as e:
        logger.error(f"Error in plotting training history: {str(e)}")
        raise


if __name__ == "__main__":
    logger.info("model_utils.py executed directly")