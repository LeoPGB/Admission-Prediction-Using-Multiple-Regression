import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

df = pd.read_csv('')

df.head()

X = df.drop('Chance of Admit ', axis =1)
y = df['Chance of Admit ']

x.head()

y.head()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 50)
reg = LinearRegression()
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)

print(metrics.r2_score(y_test,y_pred))
print(metrics.mean_squared_error(y_test,y_pred))

df_pred = pd.DataFrame({'Real Value':y_test, 'Predict Value': y_pred})

df_pred.head()

pltx = np.linspace(0.4, 1, 100)
plty = pltx
plt.scatter(y_pred, y_test)
plt.plot(pltx, plty, color='red')

