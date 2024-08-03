
# Lightweight Fine-Tuning with PEFT and DistilBERT

## Project Overview

This project demonstrates the application of Lightweight Fine-Tuning techniques using the Hugging Face `peft` library. The focus is on adapting the DistilBERT model for emotion classification using the Emotion dataset. The project highlights the effectiveness of Parameter-Efficient Fine-Tuning (PEFT) through Low-Rank Adaptation (LoRA).

## Key Features

- **Model Used**: DistilBERT (distilbert-base-uncased)
- **Fine-Tuning Technique**: LoRA (Low-Rank Adaptation) using the Hugging Face `peft` library
- **Dataset**: Emotion Dataset from Hugging Face
- **Evaluation Metric**: Accuracy

## Results

- **Pre-trained Model Accuracy**: 11.85%
- **Fine-Tuned Model Accuracy**: 59.00%
- **Accuracy Improvement**: 47.15%

## Project Files

- `LightweightFineTuning.ipynb`: The main Jupyter Notebook containing all code, explanations, and results.
- `README.md`: This file, providing an overview of the project.

## Challenges and Learnings

This project faced initial delays due to technical issues with local GPU resources and the Udacity workspace. After switching to Google Colab, the project was successfully completed, demonstrating the importance of flexibility and resourcefulness in machine learning projects.

## How to Use

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Tobiny/LightweightFineTuning.git
   ```
2. **Open the Notebook**:
   - Open the `LightweightFineTuning.ipynb` file in Jupyter Notebook or Google Colab.
3. **Run the Notebook**:
   - Follow the instructions within the notebook to load models, preprocess data, fine-tune the model, and evaluate the results.

## Future Work

- Further fine-tuning with additional epochs and hyperparameter tuning.
- Exploration of other PEFT methods or different datasets for more robust model adaptation.
- Experimentation with model architectures beyond DistilBERT.

## Repository

[Link to the repository](https://github.com/Tobiny/LightweightFineTuning)

---

*This project was completed as part of a course on Lightweight Fine-Tuning Techniques for NLP models, part of the Udacity with Cognizant Gen AI Externship Nanodegree Program.*
