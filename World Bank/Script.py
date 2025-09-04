import pandas as pd
import matplotlib.pyplot as plt

# ✅ Correct raw GitHub CSV link
url = "https://raw.githubusercontent.com/Prodigy-InfoTech/data-science-datasets/main/Task%201/data.csv"

df = pd.read_csv(url)

print(df.head())

# Gender Distribution
df['Gender'].value_counts().plot(kind='bar', color=['skyblue', 'lightgreen'])
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

# Age Distribution
df['Age'].plot(kind='hist', bins=10, color='orange', edgecolor='black')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()
