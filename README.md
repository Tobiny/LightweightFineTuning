# Heart Disease Prediction using PEFT

This project uses Parameter-Efficient Fine-Tuning (PEFT) techniques to adapt a pre-trained BERT model for heart disease prediction.

## Project Structure

- `data/`: Contains the dataset files
- `src/`: Source code for data preparation, model training, and evaluation
- `saved_models/`: Directory to store trained models
- `outputs/`: Directory to store output files (e.g., confusion matrices)

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Ensure you have the dataset file `heart_2022_no_nans.csv` in the `data/` directory

## Usage

1. Train the models: `python src/train.py`
This will train the base BERT model and the PEFT model, saving them in the `saved_models/` directory.
2. Evaluate the models: `python src/evaluate.py`
This will evaluate both the base BERT model and the PEFT model on the test set and display the results.

## Results

(To be updated after running the evaluation script)
