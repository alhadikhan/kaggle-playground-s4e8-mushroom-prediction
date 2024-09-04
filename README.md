# Kaggle Playground Series - Season 4, Episode 8: Mushroom Prediction

This repository contains the solution for the Kaggle Playground Series - Season 4, Episode 8 competition, where the task is to predict whether mushrooms are edible or poisonous based on their physical characteristics. 

## Competition Details

- **Objective**: Predict whether a mushroom is edible (`e`) or poisonous (`p`).
- **Evaluation Metric**: Matthews Correlation Coefficient (MCC)
- **Public Score**: 0.98233
- **Private Score**: 0.98200

## Dataset

The dataset provided for this competition includes features of mushrooms with labels indicating whether each mushroom is edible or poisonous. 

## Files Included

- `train.csv`: Training data with features and target labels.
- `test.csv`: Test data without target labels.
- `sample_submission.csv`: A sample submission file format.

## Installation

To get started, you can clone this repository and install the required packages. Make sure to have Python 3 installed.

## bash
git clone https://github.com/yourusername/kaggle-playground-s4e8-mushroom-prediction.git
cd kaggle-playground-s4e8-mushroom-prediction
pip install -r requirements.txt

## Usage
Preprocessing: Encode categorical features and handle missing values.
Model Training: Train a RandomForestClassifier on the processed training data.
Prediction: Generate predictions on the test data.
Submission: Create the submission file in the required format.
## Code
The code includes:

Data Preprocessing: Encoding and imputation of missing values.
Model Training: Training and evaluating a RandomForestClassifier.
Prediction and Submission: Generating predictions and preparing the submission file.
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

### Load datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

### Encoding categorical features
label_encoders = {}
for column in train_df.columns:
    if train_df[column].dtype == 'object' and column != 'class':
        le = LabelEncoder()
        train_df[column] = le.fit_transform(train_df[column].astype(str))
        label_encoders[column] = le

### Encode the target variable
le_target = LabelEncoder()
train_df['class'] = le_target.fit_transform(train_df['class'])

### Data preprocessing
train_df_temp = train_df.copy()
mean_imputer = SimpleImputer(strategy='mean')
knn_imputer = KNNImputer(n_neighbors=5)
train_df_temp['stem-width'] = knn_imputer.fit_transform(train_df_temp[['stem-width']])
train_df_temp['cap-diameter'] = mean_imputer.fit_transform(train_df_temp[['cap-diameter']])
train_df_temp.drop(columns=['id', 'stem-root', 'veil-type', 'spore-print-color', 'gill-spacing'], inplace=True)

### Splitting the data
X = train_df_temp.drop(columns=['class'])
y = train_df_temp['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

### Training the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

### Predicting and evaluating
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
## Acknowledgments
Kaggle: For organizing the Playground Series and providing the competition dataset.
Community: For sharing valuable resources and solutions.
## License
This project is licensed under the MIT License. See the LICENSE file for more details.
