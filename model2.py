import pandas as pd
from model import train_and_predict, predict_next_period_date
from sklearn.preprocessing import LabelEncoder
import datetime
import pickle
import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv('synthetic_period_data.csv')

# Check if 'Last Period Date' exists, if not, handle accordingly
if 'Last Period Date' not in df.columns:
    print("Column 'Last Period Date' not found. Using default date '2023-01-01' for all rows.")
    df['Last Period Date'] = '2023-01-01'  # Or handle with a default date

# Encode categorical columns (Symptoms, Stress Level, and Exercise Level)
label_encoder = LabelEncoder()
df['Symptoms'] = label_encoder.fit_transform(df['Symptoms'])
df['Stress Level'] = label_encoder.fit_transform(df['Stress Level'])
df['Exercise Level'] = label_encoder.fit_transform(df['Exercise Level'])

# Train the model
model = train_and_predict(df)

# Function to get user input
def get_user_input():
    # Ask for the last period date (in the format DD-MM-YYYY)
    last_period_date_input = input("Enter your last period date (DD-MM-YYYY): ")
    try:
        last_period_date = pd.to_datetime(last_period_date_input, format='%d-%m-%Y')
    except ValueError:
        print("Invalid date format. Please enter a valid date in DD-MM-YYYY format.")
        return None, None, None, None
    
    # Example encoding for symptoms, stress level, and exercise level
    symptoms_encoded = 1  # Example encoding (0: No, 1: Yes)
    stress_level_encoded = 3  # Example: 1 - Low, 2 - Medium, 3 - High
    exercise_level_encoded = 2  # Example: 1 - Low, 2 - Medium, 3 - High
    
    return last_period_date, symptoms_encoded, stress_level_encoded, exercise_level_encoded

# Get user input
last_period_date, symptoms_encoded, stress_level_encoded, exercise_level_encoded = get_user_input()

if last_period_date is None:
    print("Exiting due to invalid date input.")
    exit()

# Ask for cycle length and period duration
cycle_length = int(input("Cycle Length (in days): "))
period_duration = int(input("Period Duration (in days): "))

# Predict the next period date (the number of days to the next period)
predicted_days = predict_next_period_date(
    model, cycle_length, period_duration, symptoms_encoded, stress_level_encoded, exercise_level_encoded
)

# If predicted_days is a Timestamp, we need to extract the number of days
if isinstance(predicted_days, pd.Timestamp):
    predicted_days = (predicted_days - last_period_date).days  # Get the difference in days

# Add the cycle length to the last period date
next_period_date = last_period_date + pd.DateOffset(days=cycle_length)

# Check for valid next_period_date
if pd.isna(next_period_date):
    print("Error: Invalid next period date calculated.")
else:
    # Adjust the month and year based on the new calculated date
    if next_period_date.month != last_period_date.month:  # if the month changes
        print(f"Month transition detected: {last_period_date.month} to {next_period_date.month}")
    if next_period_date.year != last_period_date.year:  # if the year changes
        print(f"Year transition detected: {last_period_date.year} to {next_period_date.year}")

    # Print the predicted next period date
    print(f"Predicted Next Period Date: {next_period_date.strftime('%d-%m-%Y')}")

if __name__ == "__main__":
# Save model to pickle file
    with open('predictionmodel.pkl', 'wb') as model2_file:
        pickle.dump(model, model2_file)
