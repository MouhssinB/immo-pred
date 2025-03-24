from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


df = pd.read_csv("data/ValeursFoncieres-2023.txt" , sep="|" , low_memory=False)


# Encode categorical features
df_encoded = df.copy()
label_encoders = {}
categorical_columns = df_encoded.select_dtypes(include=['object']).columns

for col in categorical_columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    label_encoders[col] = le

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
