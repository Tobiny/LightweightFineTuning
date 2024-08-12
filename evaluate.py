# Import necessary libraries
from transformers import AutoTokenizer, TrainingArguments, Trainer
from peft import AutoPeftModelForSequenceClassification
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score

# Define model and tokenizer names
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the Emotion dataset from Hugging Face
dataset = load_dataset("emotion")

# Preprocessing function to tokenize and format the data
def preprocess_function(examples):
    """
    Tokenizes and formats the input examples for evaluation.

    Args:
        examples: A dictionary of examples containing text and labels.

    Returns:
        A dictionary of tokenized and formatted examples.
    """
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

# Apply the preprocessing function to the dataset
encoded_dataset = dataset.map(preprocess_function, batched=True)

# Extract the test dataset
test_dataset = encoded_dataset["test"]

# Rename the 'label' column to 'labels' and set the format for evaluation
test_dataset = test_dataset.rename_column("label", "labels")
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Define the evaluation metric: accuracy
def compute_metrics(p):
    """
    Computes the accuracy score.

    Args:
        p: A prediction object containing predicted and true labels.

    Returns:
        A dictionary containing the accuracy score.
    """
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

# Load the fine-tuned PEFT model
lora_model = AutoPeftModelForSequenceClassification.from_pretrained("lora-distilbert-emotion", num_labels=6)

# Define evaluation arguments
eval_args = TrainingArguments(
    output_dir="./results",  # Output directory for evaluation results
    per_device_eval_batch_size=16,  # Batch size for evaluation
)

# Create a Trainer instance for evaluation
trainer = Trainer(
    model=lora_model,  # Use the fine-tuned LoRA-adapted model
    args=eval_args,  # Evaluation arguments
    eval_dataset=test_dataset,  # Evaluation dataset
    compute_metrics=compute_metrics,  # Evaluation metric
)

# Evaluate the model
fine_tuned_results = trainer.evaluate()

# Print the evaluation results
print(f"Fine-tuned model accuracy: {fine_tuned_results['eval_accuracy']:.4f}")