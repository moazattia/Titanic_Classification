# 🧠 Titanic Survival Prediction — Deep Learning Project

## 🚢 Overview

> This project uses Deep Learning to predict passenger survival on the Titanic using the classic Titanic dataset

> It demonstrates data preprocessing, feature engineering, exploratory data analysis (EDA), and neural network modeling with Keras and TensorFlow.

### 📁 Project Structure

```bash
├── .env
├── .env.example
├── .gitignore
├── main.py
├── README.md
├── requirements.txt
└── src
    ├── inference.py
    ├── __init__.py
    ├── artifacts
    │   ├── best_model.keras
    │   └── preprocessor.joblib
    └── utils
        ├── config.py
        ├── request.py
        ├── response.py
        └── __init__.py

```

### 🧩 Dataset

The dataset is sourced from:

https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

Key Features:

Pclass: Passenger class (1 = Upper, 3 = Lower)

Sex: Gender

Age: Passenger age

SibSp / Parch: Family members aboard

Fare: Ticket price

Embarked: Port of embarkation

Survived: Target variable (1 = Survived, 0 = Died)

### 🔍 Exploratory Data Analysis (EDA)

EDA focused on understanding key survival factors:

Overall survival distribution

Gender and class impact on survival

Family size and alone-traveling analysis

Age and fare distribution by class

Correlation heatmap between numerical features

Example visualizations:

sns.countplot(data=df, x='sex', hue='survived')

sns.boxplot(data=df, x='pclass', y='fare', hue='survived')

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

Findings:

Females had a much higher survival rate than males.

1st class passengers survived significantly more than 3rd class.

Traveling with family slightly increased survival probability.

### ⚙️ Data Preprocessing

Steps included:

Handling missing values (Age, Embarked, Cabin)

Encoding categorical features using OneHotEncoder

Scaling numeric features using StandardScaler

Feature selection and train-test split (train_test_split)

### 🧠 Model Building

A Deep Neural Network (DNN) was built using TensorFlow & Keras.

Model Architecture:

Input layer matching feature count

Multiple Dense layers with ReLU activation

Dropout for regularization

Output layer with sigmoid activation for binary classification

model = tf.keras.Sequential([
tf.keras.layers.Dense(128, activation='relu'),
tf.keras.layers.Dropout(0.3),
tf.keras.layers.Dense(64, activation='relu'),
tf.keras.layers.Dense(1, activation='sigmoid')
])

Compilation:

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

### 🎯 Hyperparameter Tuning

Used Keras Tuner (RandomSearch) to find the best:

Number of hidden layers

Neurons per layer

Learning rate

Dropout rate

Example:

from kerastuner import RandomSearch

The best model was selected and retrained on the full training data.

### 📊 Model Evaluation

Metrics used:

Accuracy

Confusion Matrix

Precision, Recall, F1-score

ROC Curve & AUC

Visualizations like confusion matrix and ROC curve were plotted for performance insight.

### 💾 Model Saving

The trained model and preprocessing pipeline were saved for later use:

model.save('titanic_dnn_model.h5')
joblib.dump(scaler, 'scaler.pkl')

🚀 Results

The best deep learning model achieved ≈80–85% accuracy on test data.

The model effectively captured the main survival patterns: gender, class, and fare.

🧩 Future Improvements

Use Feature Engineering on the Cabin and Ticket columns.

Try ensemble methods (Random Forest, XGBoost) for comparison.

Use SMOTE or class weighting to handle data imbalance.

Deploy the model as a Streamlit web app for interactive predictions.

### 📦 Requirements

Create a requirements.txt file if you wish to replicate the environment:

tensorflow
keras-tuner
scikit-learn
matplotlib
seaborn
pandas
numpy
joblib

### Setup

1. Clone the repository
2. Create virtual environment: `python -m venv venv`
3. Activate virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and fill in the values

## Usage

Run the server:

```bash
uvicorn main:app --reload
```

#### The API will be available at `http://localhost:8000`

- Endpoints
  - GET /: Health check
  - POST /classify: Predict survival probability

### `Example Request`

```bash
[
    {
        "passenger_id": 1,
        "age": 22.0,
        "fare": 7.25,
        "sex": "male",
        "embarked": "S",
        "parch": 0,
        "sibsp": 1,
        "pclass": 3
    },
    {
        "passenger_id": 2,
        "age": 29.0,
        "fare": 15.50,
        "sex": "female",
        "embarked": "C",
        "parch": 1,
        "sibsp": 0,
        "pclass": 2
    },
    {
        "passenger_id": 3,
        "age": 35.0,
        "fare": 50.00,
        "sex": "male",
        "embarked": "Q",
        "parch": 0,
        "sibsp": 0,
        "pclass": 1
    },
    {
        "passenger_id": 4,
        "age": 18.0,
        "fare": 5.00,
        "sex": "female",
        "embarked": "S",
        "parch": 2,
        "sibsp": 3,
        "pclass": 3
    },
    {
        "passenger_id": 5,
        "age": 42.0,
        "fare": 80.00,
        "sex": "male",
        "embarked": "C",
        "parch": 1,
        "sibsp": 0,
        "pclass": 1
    }
]
```

### 👨‍💻 Author

Supervised by : Eng/Mohammed Agoor

Done by : Moaz Attia
Machine Learning Engineer
📧 moazattia58@gmail.com
