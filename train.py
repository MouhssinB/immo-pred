from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


df_in = pd.read_csv(r"data/ValeursFoncieres-2023.txt" , low_memory=False)
df_in = df_in.reset_index(drop=True)

df_encoded = df_in 
# Supprimer la colonne "Unnamed: 0" si elle existe
if "Unnamed: 0" in df_encoded.columns:
    df_encoded = df_encoded.drop(columns=["Unnamed: 0"])
print(df_encoded.head())
# Define target and features
features = df_encoded.drop(columns=["Valeur fonciere"])
target = df_encoded["Valeur fonciere"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train Random Forests
model = RandomForestRegressor(random_state=42)



model.fit(X_train, y_train)

# Get feature importances
importances = pd.Series(model.feature_importances_, index=features.columns)


# Sort and display
importances_sorted = importances.sort_values(ascending=False)


# Plotting
plt.figure(figsize=(12, 5))
sns.barplot(x=importances_sorted.values, y=importances_sorted.index)
plt.title("Feature Importance for Valeur fonciere")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

importances_sorted



import joblib

joblib.dump(model, r'data/model.pkl')