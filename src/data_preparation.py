import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE


def load_data(file_path):
    return pd.read_csv(file_path)


def preprocess_data(df):
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

    # Convert other categorical variables to numerical
    categorical_columns = ['State', 'Sex', 'GeneralHealth', 'LastCheckupTime', 'RemovedTeeth',
                           'HadDiabetes', 'SmokerStatus', 'ECigaretteUsage', 'RaceEthnicityCategory',
                           'AgeCategory', 'TetanusLast10Tdap', 'CovidPos']

    le = LabelEncoder()
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col].astype(str))

    # Normalize numerical features
    numerical_columns = ['PhysicalHealthDays', 'MentalHealthDays', 'SleepHours',
                         'HeightInMeters', 'WeightInKilograms', 'BMI']
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # Separate features and target
    X = df.drop('HadHeartAttack', axis=1)
    y = df['HadHeartAttack']

    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def handle_class_imbalance(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


def create_text_representation(df):
    text_data = []
    for _, row in df.iterrows():
        text = ""
        for col, value in row.items():
            text += f"{col}: {value}, "
        text_data.append(text[:-2])  # Remove last comma and space
    return text_data