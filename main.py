import pandas as pd
import matplotlib.pyplot as plt
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



