# ----------------------------------------
# Internship Task: Bank Marketing Decision Tree
# ----------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn import tree
import joblib

# -------------------------------
# 1️⃣ Load Dataset
# -------------------------------
data = pd.read_csv("bank-full.csv", sep=';')  # Make sure CSV is in same folder
print("First 5 rows of dataset:\n", data.head())

# -------------------------------
# 2️⃣ Check for missing values
# -------------------------------
print("\nMissing values in dataset:\n", data.isnull().sum())

# -------------------------------
# 3️⃣ Encode categorical variables
# -------------------------------
label_encoders = {}
for col in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# -------------------------------
# 4️⃣ Define features & target
# -------------------------------
X = data.drop('y', axis=1)
y = data['y']

# -------------------------------
# 5️⃣ Split data into train/test
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 6️⃣ Train Decision Tree
# -------------------------------
dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_model.fit(X_train, y_train)

# -------------------------------
# 7️⃣ Evaluate Model
# -------------------------------
y_pred = dt_model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# 8️⃣ Feature Importance
# -------------------------------
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("\nFeature Importance:\n", feature_importance)

# -------------------------------
# 9️⃣ Decision Tree Rules (Text)
# -------------------------------
tree_rules = export_text(dt_model, feature_names=list(X.columns))
print("\nDecision Tree Rules:\n", tree_rules)

# -------------------------------
# 🔟 Visualize Decision Tree (Graphical)
# -------------------------------
plt.figure(figsize=(20,10))
tree.plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=['No','Yes'],
    filled=True,
    rounded=True,
    fontsize=12
)
plt.title("Decision Tree for Customer Subscription")
plt.show()

# -------------------------------
# 1️⃣1️⃣ Save Model
# -------------------------------
joblib.dump(dt_model, 'decision_tree_model.pkl')
print("\nDecision tree model saved as 'decision_tree_model.pkl'")

# -------------------------------
# 1️⃣2️⃣ Example Prediction on New Customer
# -------------------------------
# Example: new customer data (encoded same as training)
# Adjust values as needed, number of features must match X.columns
new_customer = [[35, 2, 1, 2, 0, 500, 1, 0, 1, 5, 4, 120, 1, -1, 0, 0]]
prediction = dt_model.predict(new_customer)
print("\nWill the new customer purchase?", "Yes" if prediction[0]==1 else "No")
