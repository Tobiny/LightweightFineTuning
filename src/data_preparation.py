import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import logging
import os
from datetime import datetime


def setup_logging():
    log_dir = '../logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'data_preparation_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')

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


def load_data(file_path):
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def preprocess_data(df):
    try:
        logger.info("Starting data preprocessing")

        # Convert binary categorical variables to numerical
        binary_columns = ['PhysicalActivities', 'HadHeartAttack', 'HadAngina', 'HadStroke',
                          'HadAsthma', 'HadSkinCancer', 'HadCOPD', 'HadDepressiveDisorder',
                          'HadKidneyDisease', 'HadArthritis', 'DeafOrHardOfHearing',
                          'BlindOrVisionDifficulty', 'DifficultyConcentrating', 'DifficultyWalking',
                          'DifficultyDressingBathing', 'DifficultyErrands', 'ChestScan',
                          'AlcoholDrinkers', 'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver',
                          'HighRiskLastYear']

        for col in binary_columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
        logger.info("Binary columns converted to numerical")

        # Convert other categorical variables to numerical
        categorical_columns = ['State', 'Sex', 'GeneralHealth', 'LastCheckupTime', 'RemovedTeeth',
                               'HadDiabetes', 'SmokerStatus', 'ECigaretteUsage', 'RaceEthnicityCategory',
                               'AgeCategory', 'TetanusLast10Tdap', 'CovidPos']

        le = LabelEncoder()
        for col in categorical_columns:
            df[col] = le.fit_transform(df[col].astype(str))
        logger.info("Categorical columns encoded")

        # Normalize numerical features
        numerical_columns = ['PhysicalHealthDays', 'MentalHealthDays', 'SleepHours',
                             'HeightInMeters', 'WeightInKilograms', 'BMI']
        scaler = StandardScaler()
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
        logger.info("Numerical columns normalized")

        # Separate features and target
        X = df.drop('HadHeartAttack', axis=1)
        y = df['HadHeartAttack']
        logger.info("Features and target separated")

        return X, y
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        raise


def split_data(X, y, test_size=0.2, random_state=42):
    try:
        logger.info("Splitting data into train and test sets")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,
                                                            stratify=y)
        logger.info(f"Data split completed. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error in splitting data: {str(e)}")
        raise


def handle_class_imbalance(X, y):
    try:
        logger.info("Handling class imbalance using SMOTE")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        logger.info(f"Class imbalance handled. New shape: {X_resampled.shape}")
        return X_resampled, y_resampled
    except Exception as e:
        logger.error(f"Error in handling class imbalance: {str(e)}")
        raise


def create_text_representation(df):
    try:
        logger.info("Creating text representation of features")
        text_data = []
        for _, row in df.iterrows():
            text = ", ".join([f"{col}: {value}" for col, value in row.items()])
            text_data.append(text)
        logger.info(f"Text representation created. Number of samples: {len(text_data)}")
        return text_data
    except Exception as e:
        logger.error(f"Error in creating text representation: {str(e)}")
        raise


def prepare_data(file_path):
    try:
        df = load_data(file_path)
        X, y = preprocess_data(df)
        X_train, X_test, y_train, y_test = split_data(X, y)
        X_train_resampled, y_train_resampled = handle_class_imbalance(X_train, y_train)
        X_train_text = create_text_representation(X_train_resampled)
        X_test_text = create_text_representation(X_test)
        logger.info("Data preparation completed successfully")
        return X_train_text, X_test_text, y_train_resampled, y_test
    except Exception as e:
        logger.error(f"Error in data preparation pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        file_path = '../data/heart_2022_no_nans.csv'
        X_train_text, X_test_text, y_train, y_test = prepare_data(file_path)
        logger.info("Data preparation script executed successfully")
    except Exception as e:
        logger.error(f"Script execution failed: {str(e)}")