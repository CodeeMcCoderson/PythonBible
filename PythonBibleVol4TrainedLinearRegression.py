import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

file = 'C:/Users/dbialoncik/MachineLearning/student-mat.csv'
# data from https://archive.ics.uci.edu/ml/datasets/student+performance
# Load the data, we have to tell the program our data is separated by semicolons
data = pd.read_csv(file, sep=';')

# Determine which features may be relevant to what you want to predict and load them in
# In this data set the G's are the grades. We want to predict the third grade
# G3 is the label, the rest are features
data = data[['age', 'sex', 'studytime', 'absences', 'G1', 'G2', 'G3']]

# The sex is non-numeric, in order to work with in coordinate system we must convert
data['sex'] = data['sex'].map({'F': 0, 'M': 1})

# Define the desired label for simplicity
prediction = 'G3'

# sklearn does not accept Pandas dataframes, must turn to NumPy array
X = np.array(data.drop([prediction], 1))
Y = np.array(data[prediction])

# To train the model we must split our dataset
# We use the train data to get the hyperplane to fit our data as well as possible
# The second set of data tests the accuracy of the predictions
# 0.1 means we are using 10% of our data to train on (an industry standard)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

# Create a model and fit it with our data (this trains the model)
model = LinearRegression()
model.fit(X_train, Y_train)

# Check the accuracy (since splitting data is random, the answer may vary slightly)
accuracy = model.score(X_test, Y_test)
print(accuracy)

# Now that we know accuracy, enter new data and predict the final grade
# Data is entered with values for features in the right order (age, sex, studytime, absences, G1, G2)
X_new = np.array([[18, 1, 3, 40, 15, 16]])
Y_new = model.predict(X_new)
print(Y_new)

# That is the whole model, to visual the coorelations better we can view them. Pass (x,y)
plt.scatter(data['studytime'], data['G3'])
plt.title('Correlation')
plt.xlabel('Study Time')
plt.ylabel('Final Grade')
plt.show()
# This graphs showns almost no correlation between studytime and final grade

# You can view all correlations (copy/paste), but we will do one more
plt.scatter(data['G2'], data['G3'])
plt.title('Correlation')
plt.xlabel('Second Grade')
plt.ylabel('Final Grade')
plt.show()
# This graph shows strong correlation
