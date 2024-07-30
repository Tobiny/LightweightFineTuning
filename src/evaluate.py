import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from peft import PeftModel, PeftConfig
from data_preparation import prepare_data
from model_utils import evaluate_model, plot_confusion_matrix
import logging
from datetime import datetime
import matplotlib.pyplot as plt


def setup_logging():
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'evaluation_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

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


def load_model(model_path, device, is_peft=False):
    try:
        if is_peft:
            config = PeftConfig.from_pretrained(model_path)
            model = BertForSequenceClassification.from_pretrained(config.base_model_name_or_path)
            model = PeftModel.from_pretrained(model, model_path)
        else:
            model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
            model.load_state_dict(torch.load(model_path, map_location=device))

        model.to(device)
        model.eval()
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {str(e)}")
        raise


def evaluate_model_performance(model, dataloader, device):
    try:
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
    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        raise


def compare_models(base_metrics, peft_metrics):
    try:
        logger.info("Comparing model performances")
        metrics = base_metrics.keys()

        plt.figure(figsize=(10, 6))
        x = range(len(metrics))
        plt.bar([i - 0.2 for i in x], [base_metrics[m] for m in metrics], width=0.4, label='Base BERT', align='center')
        plt.bar([i + 0.2 for i in x], [peft_metrics[m] for m in metrics], width=0.4, label='PEFT', align='center')
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Performance Comparison: Base BERT vs PEFT')
        plt.xticks(x, metrics, rotation=45)
        plt.legend()
        plt.tight_layout()
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
        plt.close()

        logger.info("Model comparison plot saved")

        for metric in metrics:
            base_value = base_metrics[metric]
            peft_value = peft_metrics[metric]
            difference = peft_value - base_value
            logger.info(f"{metric}: Base = {base_value:.4f}, PEFT = {peft_value:.4f}, Difference = {difference:.4f}")
    except Exception as e:
        logger.error(f"Error during model comparison: {str(e)}")
        raise


def main():
    try:
        logger.info("Starting evaluation process")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        # Prepare data
        file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'heart_2022_no_nans.csv')
        _, _, X_test, _, _, y_test = prepare_data(file_path)

        # Load tokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # Tokenize data
        test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=512)

        # Create dataset and dataloader
        test_dataset = TensorDataset(
            torch.tensor(test_encodings['input_ids']),
            torch.tensor(test_encodings['attention_mask']),
            torch.tensor(y_test.values)
        )
        test_dataloader = DataLoader(test_dataset, batch_size=16)

        # Evaluate base BERT model
        base_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'saved_models',
                                       'final_base_model.pth')
        base_model = load_model(base_model_path, device)
        logger.info("Evaluating base BERT model")
        base_metrics = evaluate_model_performance(base_model, test_dataloader, device)

        # Evaluate PEFT model
        peft_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'saved_models',
                                       'final_peft_model.pth')
        peft_model = load_model(peft_model_path, device, is_peft=True)
        logger.info("Evaluating PEFT model")
        peft_metrics = evaluate_model_performance(peft_model, test_dataloader, device)

        # Compare models
        compare_models(base_metrics, peft_metrics)

        logger.info("Evaluation process completed successfully")

    except Exception as e:
        logger.error(f"Error in main evaluation process: {str(e)}")


if __name__ == "__main__":
    main()