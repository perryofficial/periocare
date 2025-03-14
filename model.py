import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve
import pickle
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Load dataset (replace with the actual dataset path)
df = pd.read_csv('synthetic_period_data.csv')

# Data Preprocessing - Encoding categorical columns
encoder = LabelEncoder()
df['Symptoms'] = encoder.fit_transform(df['Symptoms'])
df['Stress Level'] = encoder.fit_transform(df['Stress Level'])
df['Exercise Level'] = encoder.fit_transform(df['Exercise Level'])

# Define valid categories for user input
valid_symptoms = ['fatigue', 'headache', 'mood swings', 'none']
valid_stress_levels = ['low', 'medium', 'high']
valid_exercise_levels = ['low', 'medium', 'high', 'none']

# Function to encode user input
def encode_user_input(symptoms, stress_level, exercise_level):
    encoder.fit(valid_symptoms)
    symptoms_encoded = encoder.transform([symptoms.strip()])[0]  # Strip any accidental spaces

    encoder.fit(valid_stress_levels)
    stress_level_encoded = encoder.transform([stress_level.strip()])[0]

    encoder.fit(valid_exercise_levels)
    exercise_level_encoded = encoder.transform([exercise_level.strip()])[0]

    return symptoms_encoded, stress_level_encoded, exercise_level_encoded

# Get user input and validate it
def get_user_input():
    symptoms = input(f"Symptoms ({', '.join(valid_symptoms)}): ").lower().strip()
    while symptoms not in valid_symptoms:
        print(f"Invalid symptom. Please choose from {', '.join(valid_symptoms)}.")
        symptoms = input(f"Symptoms ({', '.join(valid_symptoms)}): ").lower().strip()

    stress_level = input(f"Stress Level ({', '.join(valid_stress_levels)}): ").lower().strip()
    while stress_level not in valid_stress_levels:
        print(f"Invalid stress level. Please choose from {', '.join(valid_stress_levels)}.")
        stress_level = input(f"Stress Level ({', '.join(valid_stress_levels)}): ").lower().strip()

    exercise_level = input(f"Exercise Level ({', '.join(valid_exercise_levels)}): ").lower().strip()
    while exercise_level not in valid_exercise_levels:
        print(f"Invalid exercise level. Please choose from {', '.join(valid_exercise_levels)}.")
        exercise_level = input(f"Exercise Level ({', '.join(valid_exercise_levels)}): ").lower().strip()

    symptoms_encoded, stress_level_encoded, exercise_level_encoded = encode_user_input(symptoms, stress_level, exercise_level)
    return symptoms_encoded, stress_level_encoded, exercise_level_encoded

# Feature Selection
X = df[['Cycle Length', 'Period Duration', 'Symptoms', 'Stress Level', 'Exercise Level']]  # Features
y = df['Irregular Cycle']  # Target variable

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# List of Algorithms to evaluate
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42),
    "Naive Bayes": GaussianNB(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Perceptron": Perceptron(random_state=42),
    "MLP Classifier": MLPClassifier(random_state=42)
}

# Train and evaluate each model
for name, model in models.items():
    print(f"\nTraining {name}...")

    start_time = time.time()  # Start time for execution

    # Train the model
    model.fit(X_train, y_train)

    # Predict using the trained model
    y_pred = model.predict(X_test)

    # Model Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")

    print(f"Confusion Matrix for {name}:")
    print(confusion_matrix(y_test, y_pred))

    print(f"\nClassification Report for {name}:")
    print(classification_report(y_test, y_pred))

    # ROC AUC and Precision-Recall Curve for binary classification
    if len(set(y)) == 2:  # Check if binary classification
        try:
            roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
            print(f"\nROC-AUC for {name}: {roc_auc:.2f}")

            # ROC Curve
            fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:,1])
            plt.plot(recall, precision, label=f'{name} Precision-Recall')
        except AttributeError:
            print(f"\n{name} model does not support probability-based predictions.")

    # Cross-validation score (mean accuracy across folds)
    cross_val = cross_val_score(model, X, y, cv=5)
    print(f"\nCross-Validation Accuracy for {name}: {cross_val.mean() * 100:.2f}%")

    # Execution time
    end_time = time.time()
    print(f"Execution Time for {name}: {end_time - start_time:.2f} seconds\n")

# Show ROC and Precision-Recall Curve for all models
plt.title('ROC and Precision-Recall Curves')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend(loc='best')
plt.show()

# Prediction Function for User Input
def predict_period_info(symptoms_encoded, stress_level_encoded, exercise_level_encoded):
    print("\nPlease enter the following information:")
    cycle_length = int(input("Cycle Length: "))
    period_duration = int(input("Period Duration: "))

    # Using Random Forest (or any chosen model) to make a prediction
    user_data = [[cycle_length, period_duration, symptoms_encoded, stress_level_encoded, exercise_level_encoded]]
    prediction = models["Random Forest"].predict(user_data)

    if prediction == 0:
        print("\nPrediction: Regular Cycle")
    else:
        print("\nPrediction: Irregular Cycle")

# Get user input and predict
symptoms_encoded, stress_level_encoded, exercise_level_encoded = get_user_input()
predict_period_info(symptoms_encoded, stress_level_encoded, exercise_level_encoded)

# Train and save the model for period prediction
def train_and_predict(df):
    # Handle missing 'Last Period Date' column
    if 'Last Period Date' not in df.columns:
        print("Column 'Last Period Date' not found. Using default date '2023-01-01' for all rows.")
        df['Last Period Date'] = pd.to_datetime('2023-01-01')  # Default start date

    df['Last Period Date'] = pd.to_datetime(df['Last Period Date'])

    # Calculate 'Next Period Date'
    df['Next Period Date'] = df['Last Period Date'] + pd.to_timedelta(df['Cycle Length'], unit='D')

    # Define features (Cycle Length, Period Duration, Symptoms, etc.)
    X = df[['Cycle Length', 'Period Duration', 'Symptoms', 'Stress Level', 'Exercise Level']]
    y = df['Next Period Date']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model

def predict_next_period_date(model, cycle_length, period_duration, symptoms_encoded, stress_level_encoded, exercise_level_encoded):
    # Prepare input data for prediction
    user_input = [[cycle_length, period_duration, symptoms_encoded, stress_level_encoded, exercise_level_encoded]]

    # Predict the next period date
    predicted_date = model.predict(user_input)
    predicted_date = pd.to_datetime(predicted_date[0])

    return predicted_date

# Save the trained model
with open('regularmodel.pkl', 'wb') as model1_file:
    pickle.dump(train_and_predict(df), model1_file)
