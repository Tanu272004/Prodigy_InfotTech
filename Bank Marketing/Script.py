import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset


# Make sure the path points exactly to the CSV file
file_path = r"T:\GitHub\Financial report\Prodigy_Infotech\Bank Marketing\bank-full.csv"
data = pd.read_csv(file_path, sep=';')
print(data.head())

print(data.head())


# Check the first few rows
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Encode categorical variables
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Separate features and target
X = data.drop('y', axis=1)  # Features
y = data['y']               # Target (will customer subscribe or not)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train classifier
dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Classification report
print("Classification Report:\n", classification_report(y_test, y_pred))
# Print the decision rules
tree_rules = export_text(dt_model, feature_names=list(X.columns))
print(tree_rules)


import matplotlib.pyplot as plt
from sklearn import tree

plt.figure(figsize=(20,10))
tree.plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=['No', 'Yes'],
    filled=True,
    rounded=True,
    fontsize=12
)
plt.title("Decision Tree for Customer Subscription")
plt.show()


import pandas as pd

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print(feature_importance)


import joblib

joblib.dump(dt_model, 'decision_tree_model.pkl')
# Load later with:
# dt_model = joblib.load('decision_tree_model.pkl')


import pandas as pd

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print(feature_importance)


import matplotlib.pyplot as plt
from sklearn import tree

plt.figure(figsize=(20,10))
tree.plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=['No', 'Yes'],
    filled=True,
    rounded=True,
    fontsize=12
)
plt.show()


# Example new customer data (encoded the same way as training set)
new_customer = [[35, 2, 1, 2, 0, 500, 1, 0, 1, 5, 4, 120, 1, -1, 0, 0]]
prediction = dt_model.predict(new_customer)
print("Will purchase?" , "Yes" if prediction[0] == 1 else "No")


import joblib
joblib.dump(dt_model, 'decision_tree_model.pkl')
dt_model = joblib.load('decision_tree_model.pkl')
