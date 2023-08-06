#!/usr/bin/env python
# coding: utf-8

# # Import library for read the dataset

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("Iris.csv")


# In[3]:


df


# # Data Analysis

# In[4]:


df.info()


# In[5]:


df.isnull()


# In[7]:


df.isnull().sum()


# In[8]:


print(df.shape)


# In[9]:


print(df.describe)


# In[10]:


print(df.columns)


# In[11]:


df.nunique()


# In[16]:


select_columns = df[["SepalLengthCm", "SepalWidthCm"]]

select_columns


# In[29]:


select_columns = df.loc[(df["SepalLengthCm"] > 3.3) & (df["SepalWidthCm"] > 3.3), ["SepalLengthCm", "SepalWidthCm"]]


# In[30]:


select_columns


# In[31]:


select_column = df.loc[df["SepalLengthCm"] > 2.3, "SepalLengthCm"]

select_column


# In[32]:


print(df.dtypes)


# In[33]:


df.count()


# # Visualization

# In[34]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[36]:


df.hist(bins=10, figsize=(10,8))
plt.tight_layout()
plt.show()


# In[47]:


# Step 2: Get the list of column names except for the "Species" column (assuming "Species" is the target column)
columns_to_plot = df.columns.drop("Species")

# Step 3: Create pie charts for each column
num_columns = len(columns_to_plot)
num_rows = 2  # Change this to adjust the number of rows in the subplot grid
num_cols = (num_columns + num_rows - 1) // num_rows

plt.figure(figsize=(12, 8))

for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(num_rows, num_cols, i)
    column_counts = df[column].value_counts()
    plt.pie(column_counts, labels=column_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title(f'Pie Chart of {column}')

plt.tight_layout()
plt.show()


# In[48]:


columns_to_plot = df.columns.drop("Species")

# Step 3: Create bar charts for each column
num_columns = len(columns_to_plot)
num_rows = 2  # Change this to adjust the number of rows in the subplot grid
num_cols = (num_columns + num_rows - 1) // num_rows

plt.figure(figsize=(12, 8))

for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(num_rows, num_cols, i)
    column_counts = df[column].value_counts()
    column_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f'Bar Chart of {column}')

plt.tight_layout()
plt.show()


# In[49]:


columns_to_plot = df.columns.drop("Species")

# Step 3: Create scatter plots for each column against the "Species" column
num_columns = len(columns_to_plot)
num_rows = 2  # Change this to adjust the number of rows in the subplot grid
num_cols = (num_columns + num_rows - 1) // num_rows

plt.figure(figsize=(12, 8))

for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(num_rows, num_cols, i)
    plt.scatter(df[column], df["Species"], alpha=0.7, edgecolors='w')
    plt.xlabel(column)
    plt.ylabel("Species")
    plt.title(f'Scatter Plot: {column} vs Species')

plt.tight_layout()
plt.show()


# In[52]:


plt.figure(figsize=(8, 6))
plt.scatter(df["SepalLengthCm"], df["SepalWidthCm"], c='red', label='Sepal Length (cm)')
plt.scatter(df["SepalLengthCm"], df["SepalWidthCm"], c='yellow', label='Petal Length (cm)')
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.legend()
plt.show()


# In[53]:


plt.figure(figsize=(8, 6))
plt.scatter(df["PetalLengthCm"], df["PetalWidthCm"], c='red', label='Petal Length (cm)')
plt.scatter(df["PetalLengthCm"], df["PetalWidthCm"], c='yellow', label='Petal Width (cm)')
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.legend()
plt.show()


# In[54]:


plt.figure(figsize=(8, 6))
avg_sepal_length = df.groupby("Species")["SepalLengthCm"].mean()
sns.barplot(x=avg_sepal_length.index, y=avg_sepal_length.values, palette='pastel')
plt.xlabel("Species")
plt.ylabel("Average Sepal Length (cm)")
plt.title("Bar Plot: Average Sepal Length for Each Species")
plt.show()


# In[58]:


numerical_columns = df.drop("Species", axis=1).columns

# Step 3: Create box plots for each numerical column
num_columns = len(numerical_columns)
num_rows = 2  # Change this to adjust the number of rows in the subplot grid
num_cols = (num_columns + num_rows - 1) // num_rows

plt.figure(figsize=(12, 8))

for i, column in enumerate(numerical_columns, 1):
    plt.subplot(num_rows, num_cols, i)
    sns.boxplot(x=column, data=df, color='lightgreen')
    plt.xlabel(column)
    plt.title(f'Box Plot: Spread of {column}')

plt.tight_layout()
plt.show()


# # Machine Learning Model

# In[59]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the data and split into features (X) and target variable (y)
df = pd.read_csv("Iris.csv")
X = df.drop("Species", axis=1)
y = df["Species"]

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train multiple machine learning models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC()
}

for model_name, model in models.items():
    model.fit(X_train, y_train)

    # Step 4: Evaluate the models on the testing data
    y_pred = model.predict(X_test)

    # Step 5: Compare the performance of different models
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.2f}\n")
    print(classification_report(y_test, y_pred))
    print("=" * 50)


# In[ ]:




