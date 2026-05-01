import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv('heart.csv')
target_col = 'HeartDisease'

encoders = {}
df_encoded = df.copy()
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df_encoded.drop(target_col, axis=1)
y = df_encoded[target_col]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

models = {
    "Logistic Regression": LogisticRegression().fit(X_scaled, y),
    "Naive Bayes": GaussianNB().fit(X_scaled, y),
    "SVM": SVC().fit(X_scaled, y)
}

def start_prediction_interface():
    print("\n" + "="*30)
    print(" HEART DISEASE PREDICTOR ")
    print("="*30)
    
    user_data = {}
    
    for col in X.columns:
        if col in encoders:
            valid_options = list(encoders[col].classes_)
            while True:
                val = input(f"Enter {col} ({'/'.join(map(str, valid_options))}): ").strip()
                if val in valid_options:
                    user_data[col] = val
                    break
                print(f"❌ Invalid choice. Please enter one of: {valid_options}")
        else:
            while True:
                try:
                    val = float(input(f"Enter {col} (number): "))
                    user_data[col] = val
                    break
                except ValueError:
                    print("❌ Invalid input. Please enter a numerical value.")

    input_df = pd.DataFrame([user_data])
    for col, le in encoders.items():
        input_df[col] = le.transform(input_df[col])
    
    input_scaled = scaler.transform(input_df)

    print("\n" + "-"*30)
    print("DIAGNOSIS RESULTS:")
    for name, model in models.items():
        prediction = model.predict(input_scaled)[0]
        status = "⚠️ HEART DISEASE DETECTED" if prediction == 1 else "✅ NORMAL"
        print(f"{name:20}: {status}")
    print("-"*30)

if __name__ == "__main__":
    while True:
        start_prediction_interface()
        choice = input("\nWould you like to test another patient? (y/n): ").lower()
        if choice != 'y':
            print("Goodbye!")
            break