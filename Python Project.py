# ==============================
# IMPORT LIBRARIES
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

plt.rcParams['figure.figsize'] = (10, 6)


# LOAD DATASET

df = pd.read_csv(r"C:\Users\Utkarsh Bhardwaj\UT\Global_Space_Exploration_Dataset.csv")

# BASIC EXPLORATION
print("Columns in dataset:\n", df.columns)

print("\nFirst 5 rows:\n", df.head())

print("\nDataset Info:\n")
print(df.info())

print("\nStatistical Summary:\n", df.describe())

print("\nMissing Values:\n", df.isnull().sum())


# DATA CLEANING

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# 1. MISSIONS PER YEAR (BAR)
missions_per_year = df['Year'].value_counts().sort_index()

plt.figure()
missions_per_year.plot(kind='bar')
plt.title("Number of Missions per Year")
plt.xlabel("Year")
plt.ylabel("Count")
plt.show()


import matplotlib.pyplot as plt

# Group by Country and calculate average success rate
country_success = df.groupby('Country')['Success Rate (%)'].mean()

# Create pie chart
plt.figure(figsize=(8, 8))
plt.pie(country_success, 
        labels=None,  # we'll use legend instead
        autopct='%1.1f%%', 
        startangle=90)

# Add legend with country names
plt.legend(country_success.index, title="Countries", loc="best")

plt.title("Success Rate Distribution by Country")
plt.show()

# 3. DONUT CHART
df['Success Category'] = df['Success Rate (%)'].apply(
    lambda x: 'High Success' if x >= 75 else 'Low Success'
)

category_counts = df['Success Category'].value_counts()
plt.figure()

wedges, texts, autotexts = plt.pie(
    category_counts,
    labels=category_counts.index,
    autopct='%1.1f%%',
    startangle=90
)

centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title("Mission Success Distribution (Donut Chart)")
plt.axis('equal')
plt.show()


# 4. TOP COUNTRIES (BAR)

top_countries = df['Country'].value_counts().head(10)

plt.figure()
top_countries.plot(kind='bar')
plt.title("Top 10 Countries by Missions")
plt.ylabel("Number of Missions")
plt.show()


# 5. SCATTER PLOT (BUDGET vs SUCCESS RATE)

plt.figure()
sns.scatterplot(x='Budget (in Billion $)', y='Success Rate (%)', data=df)
plt.title("Budget vs Success Rate")
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))

sns.scatterplot(
    x='Budget (in Billion $)',   # numeric
    y='Success Rate (%)',         # numeric
    hue='Satellite Type',         # categorical (color-coded)
    data=df,
    s=100
)

plt.title("Budget vs Success Rate (by Satellite Type)")
plt.xlabel("Budget (in Billion $)")
plt.ylabel("Success Rate (%)")
plt.legend(title="Satellite Type")

plt.show()


# 7. HEATMAP (CORRELATION)
numeric_cols = df.select_dtypes(include=np.number)

plt.figure()
sns.heatmap(numeric_cols.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()


# 8. BOXPLOT (BUDGET)

plt.figure()
sns.boxplot(x=df['Budget (in Billion $)'])
plt.title("Budget Distribution")
plt.show()


# 9. PAIRPLOT

sns.pairplot(df[['Budget (in Billion $)', 'Duration (in Days)', 'Success Rate (%)']])
plt.show()


# 10. MACHINE LEARNING (REGRESSION MODEL)

X = df[['Budget (in Billion $)', 'Duration (in Days)']]
y = df['Success Rate (%)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluation
print("\nModel Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


# 11. ACTUAL vs PREDICTED

plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Success Rate")
plt.ylabel("Predicted Success Rate")
plt.title("Actual vs Predicted Success Rate")
plt.show()

0
# 12. FEATURE IMPORTANCE
importance = model.coef_

plt.figure()
plt.bar(X.columns, importance)
plt.title("Feature Importance")
plt.show()
#13count plot
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure()
sns.countplot(x='Satellite Type', data=df)
plt.title("Number of Missions by Satellite Type")
plt.xticks(rotation=45)
plt.show()
#14grouped count plot
plt.figure()
sns.countplot(x='Country', hue='Satellite Type', data=df)
plt.title("Satellite Type Distribution Across Countries")
plt.xticks(rotation=45)
plt.show()

#15stacked bar chart
cross_tab = pd.crosstab(df['Country'], df['Success Category'])

cross_tab.plot(kind='bar', stacked=True)
plt.title("Success Category by Country")
plt.xticks(rotation=45)
plt.show()
#16 pie chart
sat_type = df['Satellite Type'].value_counts()

plt.figure()
plt.pie(sat_type, labels=sat_type.index, autopct='%1.1f%%')
plt.title("Satellite Type Distribution")
plt.show()

#17 TreeMap
import squarify

sizes = df['Country'].value_counts()

plt.figure()
squarify.plot(sizes=sizes.values, label=sizes.index)
plt.title("Mission Distribution by Country")
plt.axis('off')
plt.show()
