import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# Gets dataframe from the source csv local file
df = pd.read_csv('./Salary_dataset.csv')

# Checking the quality of the data
print(df.isnull().values.any())
print(df.dtypes)
print(df.info())

# Drop a column that is not relevant for the training
df = df.drop('Unnamed: 0', axis=1)

# Check the 2 columns that we'll use
print(df.columns)
x = df.YearsExperience
y = df.Salary

# Made a plot to have a preview of the correlation and if is positive or negative
fig, ax = plt.subplots()
ax.scatter(x,y)
plt.show()

X = x.to_frame()
print(X)
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)

print(x_train)
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)


fig2, ax = plt.subplots()
ax.scatter(x_test,y_test)
plt.show()












