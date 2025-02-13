# Import necessary libraries
import tkinter as tk
from tkinter import ttk
import ttkbootstrap as tb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Creating a synthetic dataset with more real-world parameters
np.random.seed(42)
data_size = 1000
data = {
    'Industry': np.random.choice(['Tech', 'Healthcare', 'Finance', 'Retail', 'Education'], data_size),
    'Team_Size': np.random.randint(1, 50, data_size),
    'Traction': np.random.choice(['Low', 'Medium', 'High'], data_size),
    'Product_Innovation': np.random.choice(['Low', 'Medium', 'High'], data_size),
    'Revenue_Model': np.random.choice(['Subscription', 'Ad-based', 'Freemium'], data_size),
    'Market_Size': np.random.choice(['Small', 'Medium', 'Large'], data_size),
    'Business_Model': np.random.choice(['B2B', 'B2C', 'B2B2C'], data_size),
    'Team_Experience': np.random.choice(['Beginner', 'Intermediate', 'Expert'], data_size),
    'Competitive_Landscape': np.random.choice(['Low', 'Moderate', 'High'], data_size),
    'Investment_Stage': np.random.choice(['Seed', 'Series A', 'Series B'], data_size),
    'Geographic_Location': np.random.choice(['North America', 'Europe', 'Asia', 'Other'], data_size),
    'Funding_Success': np.random.choice([0, 1], data_size, p=[0.6, 0.4])  # 0 = No Funding, 1 = Funding Secured
}

df = pd.DataFrame(data)

# Step 2: Data Preprocessing
# Encoding categorical features
df_encoded = pd.get_dummies(df, columns=[
    'Industry', 'Traction', 'Product_Innovation', 'Revenue_Model', 'Market_Size',
    'Business_Model', 'Team_Experience', 'Competitive_Landscape', 
    'Investment_Stage', 'Geographic_Location'
], drop_first=True)

# Splitting features and target
X = df_encoded.drop('Funding_Success', axis=1)
y = df_encoded['Funding_Success']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 3: Model Training
# Creating and training the model
model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
model.fit(X_train, y_train)

# Step 4: Function to make predictions based on UI inputs
def predict_funding():
    new_data = {
        'Team_Size': int(team_size_var.get()),
        'Industry_Healthcare': [1 if industry_var.get() == 'Healthcare' else 0],
        'Industry_Tech': [1 if industry_var.get() == 'Tech' else 0],
        'Industry_Retail': [1 if industry_var.get() == 'Retail' else 0],
        'Industry_Finance': [1 if industry_var.get() == 'Finance' else 0],
        'Traction_Low': [1 if traction_var.get() == 'Low' else 0],
        'Traction_Medium': [1 if traction_var.get() == 'Medium' else 0],
        'Product_Innovation_Low': [1 if product_innovation_var.get() == 'Low' else 0],
        'Product_Innovation_Medium': [1 if product_innovation_var.get() == 'Medium' else 0],
        'Revenue_Model_Ad-based': [1 if revenue_model_var.get() == 'Ad-based' else 0],
        'Revenue_Model_Freemium': [1 if revenue_model_var.get() == 'Freemium' else 0],
        'Market_Size_Medium': [1 if market_size_var.get() == 'Medium' else 0],
        'Market_Size_Large': [1 if market_size_var.get() == 'Large' else 0],
        'Business_Model_B2C': [1 if business_model_var.get() == 'B2C' else 0],
        'Business_Model_B2B2C': [1 if business_model_var.get() == 'B2B2C' else 0],
        'Team_Experience_Intermediate': [1 if team_experience_var.get() == 'Intermediate' else 0],
        'Team_Experience_Expert': [1 if team_experience_var.get() == 'Expert' else 0],
        'Competitive_Landscape_Moderate': [1 if competition_var.get() == 'Moderate' else 0],
        'Competitive_Landscape_High': [1 if competition_var.get() == 'High' else 0],
        'Investment_Stage_Series A': [1 if investment_stage_var.get() == 'Series A' else 0],
        'Investment_Stage_Series B': [1 if investment_stage_var.get() == 'Series B' else 0],
        'Geographic_Location_Europe': [1 if location_var.get() == 'Europe' else 0],
        'Geographic_Location_Asia': [1 if location_var.get() == 'Asia' else 0],
        'Geographic_Location_Other': [1 if location_var.get() == 'Other' else 0],
    }

    # Create a DataFrame with the new input data
    new_data_df = pd.DataFrame(new_data)

    # Reindex the new data to ensure it has the same columns as the training data, filling missing columns with 0
    new_data_df = new_data_df.reindex(columns=X.columns, fill_value=0)

    # Scale the new data using the trained scaler
    new_data_scaled = scaler.transform(new_data_df)

    # Make the prediction using the trained model
    prediction = model.predict(new_data_scaled)
    probability = model.predict_proba(new_data_scaled)[0][1] * 100

    result_text.set(f"Funding Probability: {'Secured' if prediction[0] == 1 else 'Not Secured'} ({probability:.2f}%)")
    plot_probability_chart(probability)

def plot_probability_chart(probability):
    labels = ['Not Secured', 'Secured']
    probs = [100 - probability, probability]
    plt.figure(figsize=(6, 4))
    sns.barplot(x=labels, y=probs, palette='coolwarm')
    plt.title('Funding Probability', fontsize=14)
    plt.ylabel('Probability (%)')
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()

# Step 5: Setting up the UI
app = tb.Window(themename="darkly")
app.title("Funding Probability Predictor")

# Create input fields
ttk.Label(app, text="Team Size:").grid(row=0, column=0, padx=10, pady=5)
team_size_var = tk.StringVar()
ttk.Entry(app, textvariable=team_size_var).grid(row=0, column=1, padx=10, pady=5)

ttk.Label(app, text="Industry:").grid(row=1, column=0, padx=10, pady=5)
industry_var = tk.StringVar()
industry_combo = ttk.Combobox(app, textvariable=industry_var, values=['Tech', 'Healthcare', 'Finance', 'Retail', 'Education'])
industry_combo.grid(row=1, column=1, padx=10, pady=5)

ttk.Label(app, text="Traction:").grid(row=2, column=0, padx=10, pady=5)
traction_var = tk.StringVar()
traction_combo = ttk.Combobox(app, textvariable=traction_var, values=['Low', 'Medium', 'High'])
traction_combo.grid(row=2, column=1, padx=10, pady=5)

ttk.Label(app, text="Product Innovation:").grid(row=3, column=0, padx=10, pady=5)
product_innovation_var = tk.StringVar()
product_innovation_combo = ttk.Combobox(app, textvariable=product_innovation_var, values=['Low', 'Medium', 'High'])
product_innovation_combo.grid(row=3, column=1, padx=10, pady=5)

ttk.Label(app, text="Revenue Model:").grid(row=4, column=0, padx=10, pady=5)
revenue_model_var = tk.StringVar()
revenue_model_combo = ttk.Combobox(app, textvariable=revenue_model_var, values=['Subscription', 'Ad-based', 'Freemium'])
revenue_model_combo.grid(row=4, column=1, padx=10, pady=5)

ttk.Label(app, text="Market Size:").grid(row=5, column=0, padx=10, pady=5)
market_size_var = tk.StringVar()
market_size_combo = ttk.Combobox(app, textvariable=market_size_var, values=['Small', 'Medium', 'Large'])
market_size_combo.grid(row=5, column=1, padx=10, pady=5)

ttk.Label(app, text="Business Model:").grid(row=6, column=0, padx=10, pady=5)
business_model_var = tk.StringVar()
business_model_combo = ttk.Combobox(app, textvariable=business_model_var, values=['B2B', 'B2C', 'B2B2C'])
business_model_combo.grid(row=6, column=1, padx=10, pady=5)

ttk.Label(app, text="Team Experience:").grid(row=7, column=0, padx=10, pady=5)
team_experience_var = tk.StringVar()
team_experience_combo = ttk.Combobox(app, textvariable=team_experience_var, values=['Beginner', 'Intermediate', 'Expert'])
team_experience_combo.grid(row=7, column=1, padx=10, pady=5)

ttk.Label(app, text="Competitive Landscape:").grid(row=8, column=0, padx=10, pady=5)
competition_var = tk.StringVar()
competition_combo = ttk.Combobox(app, textvariable=competition_var, values=['Low', 'Moderate', 'High'])
competition_combo.grid(row=8, column=1, padx=10, pady=5)

ttk.Label(app, text="Investment Stage:").grid(row=9, column=0, padx=10, pady=5)
investment_stage_var = tk.StringVar()
investment_stage_combo = ttk.Combobox(app, textvariable=investment_stage_var, values=['Seed', 'Series A', 'Series B'])
investment_stage_combo.grid(row=9, column=1, padx=10, pady=5)

ttk.Label(app, text="Geographic Location:").grid(row=10, column=0, padx=10, pady=5)
location_var = tk.StringVar()
location_combo = ttk.Combobox(app, textvariable=location_var, values=['North America', 'Europe', 'Asia', 'Other'])
location_combo.grid(row=10, column=1, padx=10, pady=5)

# Prediction button
ttk.Button(app, text="Predict Funding Probability", command=predict_funding).grid(row=11, column=0, columnspan=2, padx=10, pady=10)

# Result display
result_text = tk.StringVar()
ttk.Label(app, textvariable=result_text, font=("Arial", 12)).grid(row=12, column=0, columnspan=2, padx=10, pady=10)

# Start the application
app.mainloop()
