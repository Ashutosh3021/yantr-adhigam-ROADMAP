import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression as lr
from sklearn.preprocessing import StandardScaler as ss
from sklearn.metrics import mean_squared_error, r2_score
from mlxtend.plotting import plot_decision_regions
df = pd.read_csv("Data/realistic_placement_data.csv")
x = df.iloc[:, 0:2]
y = df.iloc[:, -1]

x_train, x_test, y_train, y_test = tts(x, y, test_size=0.1)

scaler = ss()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

clf = lr()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)

plot_decision_regions(x_train,y_train.values,clf=clf,legend=2)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Regions for Linear Regression')
plt.show()