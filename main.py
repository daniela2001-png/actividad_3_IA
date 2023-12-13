# Imports the necessary libraries: pandas, matplotlib.pyplot, train_test_split from sklearn.model_selection, and LinearRegression from sklearn.linear_model
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Reads a CSV file ('Salary_dataset.csv') into a pandas DataFrame.
df = pd.read_csv(
    'https://www.kaggle.com/daniela2001/salary-dataset-simple-linear-regression'
)

# Checks the quality of the data by printing whether there are any missing values, the data types of each column, and a summary of the DataFrame.
print(df.isnull().values.any())
print(df.dtypes)
print(df.info())

# Drops a column ('Unnamed: 0') that is not relevant for the training.
df = df.drop('Unnamed: 0', axis=1)

# Prints the column names of the DataFrame and assigns the 'YearsExperience' column to variable x and the 'Salary' column to variable y.
print(df.columns)
x = df.YearsExperience
y = df.Salary

# Creates a scatter plot to visualize the correlation between 'YearsExperience' and 'Salary'.
fig, ax = plt.subplots()
ax.scatter(x, y)
plt.show()

# Converts the 'YearsExperience' column to a DataFrame called X.
X = x.to_frame()
print(X)

# Splits the data into training and test sets using train_test_split, with 23% of the data allocated for testing and a random state of 50.
x_train, x_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.23,
                                                    random_state=50)
print(x_train)

# Creates an instance of the LinearRegression model and fits it to the training data.
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicts the salary for the test set.
y_pred = regressor.predict(x_test)
"""
# Plots the predicted salary with the test data on a scatter plot, along with a line representing the regression model.
plt.scatter(df['YearsExperience'], df['Salary'], color='gold')
plt.plot(x_train, regressor.predict(x_train))
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')


corr_pearson = df.corr(method='pearson')
print(corr_pearson)
"""
