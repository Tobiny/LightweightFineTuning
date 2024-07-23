from transformers import BertTokenizer, BertForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType

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