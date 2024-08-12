# Lightweight Fine-Tuning with PEFT and DistilBERT for Emotion Classification

This repository demonstrates the application of Lightweight Fine-Tuning techniques using the Hugging Face `peft` library. The focus is on adapting the DistilBERT model for emotion classification using the Emotion dataset. The project highlights the effectiveness of Parameter-Efficient Fine-Tuning (PEFT) through Low-Rank Adaptation (LoRA).

## Key Features

- **Model Used:** DistilBERT (distilbert-base-uncased)
- **Fine-Tuning Technique:** LoRA (Low-Rank Adaptation) using the Hugging Face `peft` library
- **Dataset:** Emotion Dataset from Hugging Face
- **Evaluation Metric:** Accuracy

## Results

- **Initial Accuracy (1 Epoch, lora_alpha=16):** 0.5970
- **Improved Accuracy (3 Epochs, lora_alpha=32):** 0.7995

## Project Files

- `train.py`: The script for training the model with LoRA.
- `evaluate.py`: The script for evaluating the fine-tuned model.
- `requirements.txt`: Lists the required Python packages.

## How to Use

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Tobiny/LightweightFineTuning.git
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv .venv
   ```

3. **CActivate the Virtual Environment:**
   ```bash
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On macOS/Linux
   ```

4. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Training Script:**
   ```bash
   python train.py
   ```

6. **Run the Evaluation Script (Optional):**
   ```bash
   python evaluate.py
   ```


## Experiments and Improvements
This project involved two main experiments to improve the model's performance:

1. Increasing Training Epochs: The number of training epochs was increased from 1 to 3.
2. Adjusting LoRA Alpha: The lora_alpha parameter in the LoRA configuration was changed from 16 to 32.

These changes resulted in a significant improvement in accuracy, as shown in the Results section.

## Future Work

- Further Hyperparameter Tuning: Experiment with other hyperparameters, such as learning rate, batch size, and the LoRA r parameter.
- Data Augmentation: Explore techniques to augment the training data.
- Different PEFT Methods: Try other PEFT methods like prompt tuning or prefix tuning.
- Model Architecture: Consider using a larger base model or a different architecture.

## Conclusion
This project demonstrates the effectiveness of lightweight fine-tuning techniques, specifically LoRA, in achieving substantial improvements in accuracy with minimal computational overhead. The results highlight the potential of PEFT methods for adapting large language models to specific tasks efficiently.

---

*This project was completed as part of a course on Lightweight Fine-Tuning Techniques for NLP models, part of the Udacity with Cognizant Gen AI Externship Nanodegree Program.*