from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from transformers import BertTokenizer, BertForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def select_features(X, y, k=20):
    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_feature_indices = selector.get_support(indices=True)
    return X_selected, selected_feature_indices


def evaluate_model(y_true, y_pred, y_pred_proba=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted'),
        "recall": recall_score(y_true, y_pred, average='weighted'),
        "f1_score": f1_score(y_true, y_pred, average='weighted'),
    }
    if y_pred_proba is not None:
        metrics["auc_roc"] = roc_auc_score(y_true, y_pred_proba, average='weighted', multi_class='ovr')

    cm = confusion_matrix(y_true, y_pred)

    return metrics, cm


def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('../outputs/confusion_matrix.png')
    plt.close()


def load_bert_model(num_labels=2):
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return tokenizer, model


def create_peft_model(model, peft_config):
    return get_peft_model(model, peft_config)


def get_lora_config():
    return LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
    )