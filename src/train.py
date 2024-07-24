import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.ensemble import GradientBoostingClassifier
from data_preparation import load_data, preprocess_data, split_data, handle_class_imbalance, create_text_representation
from model_utils import select_features, evaluate_model, load_bert_model, create_peft_model, get_lora_config
from tqdm import tqdm
import joblib

def check_and_create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")

def train_gradient_boosting():
    print("Loading and preprocessing data...")
    df = load_data('../data/heart_2022_no_nans.csv')
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Handling class imbalance...")
    X_train_resampled, y_train_resampled = handle_class_imbalance(X_train, y_train)

    print("Selecting features...")
    X_train_selected, selected_feature_indices = select_features(X_train_resampled, y_train_resampled)
    X_test_selected = X_test.iloc[:, selected_feature_indices]

    print("Training Gradient Boosting model...")
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train_selected, y_train_resampled)

    print("Evaluating Gradient Boosting model...")
    y_pred = model.predict(X_test_selected)
    y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
    metrics = evaluate_model(y_test, y_pred, y_pred_proba)

    print("Gradient Boosting Model performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    return model, selected_feature_indices

def train_bert_model(model, train_dataloader, val_dataloader, device, num_epochs=3):
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(num_epochs):
        print(f"Training epoch {epoch + 1}/{num_epochs}...")
        model.train()
        train_progress = tqdm(train_dataloader, desc="Training", leave=False)
        for batch in train_progress:
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}

            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        print(f"Evaluating after epoch {epoch + 1}...")
        model.eval()
        val_preds, val_true = [], []
        val_progress = tqdm(val_dataloader, desc="Evaluating", leave=False)
        for batch in val_progress:
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}

            with torch.no_grad():
                outputs = model(**inputs)

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_true.extend(batch[2].cpu().numpy())

        metrics = evaluate_model(val_true, val_preds)

        print(f"Epoch {epoch + 1}/{num_epochs} metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

def main():
    # Verificar y crear directorios necesarios
    check_and_create_directory('../saved_models')

    print("Training Gradient Boosting model:")
    gb_model, selected_features = train_gradient_boosting()

    print("\nPreparing data for BERT model:")
    df = load_data('../data/heart_2022_no_nans.csv')
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Creating text representations...")
    X_train_text = create_text_representation(X_train)
    X_test_text = create_text_representation(X_test)

    print("Loading BERT model and tokenizer...")
    tokenizer, bert_model = load_bert_model()

    print("Tokenizing data...")
    train_encodings = tokenizer(X_train_text, truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(X_test_text, truncation=True, padding=True, max_length=512)

    print("Creating datasets...")
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

    print("Creating dataloaders...")
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    bert_model.to(device)

    print("\nTraining base BERT model:")
    train_bert_model(bert_model, train_dataloader, test_dataloader, device)

    print("\nCreating and training PEFT model:")
    peft_config = get_lora_config()
    peft_model = create_peft_model(bert_model, peft_config)
    peft_model.to(device)

    train_bert_model(peft_model, train_dataloader, test_dataloader, device)

    # Save the models
    gb_model_path = '../saved_models/gradient_boosting_model.joblib'
    bert_model_path = '../saved_models/bert_model'
    peft_model_path = '../saved_models/peft_model'

    print("Saving models...")
    try:
        joblib.dump(gb_model, gb_model_path)
        print(f"Gradient Boosting model saved to {gb_model_path}")
    except Exception as e:
        print(f"Error saving Gradient Boosting model: {e}")

    try:
        bert_model.save_pretrained(bert_model_path)
        print(f"BERT model saved to {bert_model_path}")
    except Exception as e:
        print(f"Error saving BERT model: {e}")

    try:
        peft_model.save_pretrained(peft_model_path)
        print(f"PEFT model saved to {peft_model_path}")
    except Exception as e:
        print(f"Error saving PEFT model: {e}")

    print("\nTraining and evaluation complete.")

if __name__ == "__main__":
    main()