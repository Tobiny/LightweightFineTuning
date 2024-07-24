import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer
from peft import PeftModel, PeftConfig
from data_preparation import load_data, preprocess_data, split_data, create_text_representation
from model_utils import evaluate_model, load_bert_model, plot_confusion_matrix


def load_peft_model(base_model_name, peft_model_path):
    config = PeftConfig.from_pretrained(peft_model_path)
    model = PeftModel.from_pretrained(base_model_name, peft_model_path)
    return model


def main():
    print("Loading and preprocessing data...")
    df = load_data('../data/heart_2022_no_nans.csv')
    X, y = preprocess_data(df)
    _, X_test, _, y_test = split_data(X, y)

    print("Creating text representations...")
    X_test_text = create_text_representation(X_test)

    print("Loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    print("Tokenizing data...")
    test_encodings = tokenizer(X_test_text, truncation=True, padding=True, max_length=512)

    print("Creating dataset...")
    test_dataset = TensorDataset(
        torch.tensor(test_encodings['input_ids']),
        torch.tensor(test_encodings['attention_mask']),
        torch.tensor(y_test.values)
    )

    print("Creating dataloader...")
    test_dataloader = DataLoader(test_dataset, batch_size=16)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\nEvaluating base BERT model:")
    _, base_model = load_bert_model()
    base_model.to(device)
    base_model.eval()

    print("\nEvaluating PEFT model:")
    peft_model = load_peft_model("bert-base-uncased", "../saved_models/peft_model")
    peft_model.to(device)
    peft_model.eval()

    models = [("Base BERT", base_model), ("PEFT BERT", peft_model)]

    for model_name, model in models:
        print(f"\nEvaluating {model_name}:")
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_dataloader:
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

            print(f"{model_name} performance:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()