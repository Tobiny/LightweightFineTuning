# Import necessary libraries
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import numpy as np
from sklearn.metrics import accuracy_score

# Define model and tokenizer names
model_name = "distilbert-base-uncased"  # Use the pre-trained DistilBERT model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the Emotion dataset from Hugging Face
dataset = load_dataset("emotion")

# Preprocessing function to tokenize and format the data
def preprocess_function(examples):
    """
    Tokenizes and formats the input examples for training.
    
    Args:
        examples: A dictionary of examples containing text and labels.
        
    Returns:
        A dictionary of tokenized and formatted examples.
    """
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

# Apply the preprocessing function to the dataset
encoded_dataset = dataset.map(preprocess_function, batched=True)

# Split the dataset into train and test sets
train_dataset = encoded_dataset["train"]
test_dataset = encoded_dataset["test"]

# Rename the 'label' column to 'labels' and set the format for training
train_dataset = train_dataset.rename_column("label", "labels")
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
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

# Load the pre-trained DistilBERT model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)

# Create a LoRA configuration for Parameter-Efficient Fine-Tuning
lora_config = LoraConfig(
    r=8,  # Rank of the low-rank matrices
    lora_alpha=16,  # Scaling factor for the LoRA updates
    target_modules=["q_lin", "v_lin", "k_lin", "out_lin"],  # Target specific linear layers in DistilBERT
    lora_dropout=0.1,  # Dropout probability for the LoRA layers
    bias="none",  # No bias term for the LoRA layers
    task_type="SEQ_CLS"  # Sequence classification task
)

# Apply LoRA to the pre-trained model
lora_model = get_peft_model(model, lora_config)

# Print the number of trainable parameters
lora_model.print_trainable_parameters()

# Define training arguments
training_args = TrainingArguments(
    output_dir="./lora-distilbert-emotion",  # Output directory for the trained model
    evaluation_strategy="epoch",  # Evaluate the model at the end of each epoch
    learning_rate=2e-5,  # Learning rate for the optimizer
    per_device_train_batch_size=16,  # Batch size for training
    per_device_eval_batch_size=16,  # Batch size for evaluation
    num_train_epochs=1,  # Number of training epochs
    weight_decay=0.01,  # Weight decay for regularization
    save_steps=10_000,  # Save the model every 10,000 steps
    save_total_limit=2,  # Keep only the last 2 saved checkpoints
    logging_dir="./logs",  # Directory for logging training information
)

# Create a Trainer instance for fine-tuning
trainer = Trainer(
    model=lora_model,  # Use the LoRA-adapted model
    args=training_args,  # Training arguments
    train_dataset=train_dataset,  # Training dataset
    eval_dataset=test_dataset,  # Evaluation dataset
    compute_metrics=compute_metrics,  # Evaluation metric
)

# Train the model
trainer.train()

# Save the fine-tuned model
lora_model.save_pretrained("lora-distilbert-emotion")

print("Training completed. Model saved to 'lora-distilbert-emotion'.")