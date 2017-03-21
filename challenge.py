import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

#read data
dataframe = pd.read_csv('challenge_dataset.txt')
x_values = dataframe.ix[:,0]
y_values = dataframe.ix[:,1]

#train model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values.reshape(-1, 1), y_values.reshape(-1, 1))

#visualize results
plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values.reshape(-1, 1)))
plt.show()
