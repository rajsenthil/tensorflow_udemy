# Data Visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.arange(0,10)
y = x**2
plt.plot(x,y, 'r--')
plt.xlim(0,4)
plt.ylim(0,10)
plt.title("PLOTTING")
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.show()

mat = np.arange(0,100).reshape(10,10)
plt.imshow(mat, cmap='coolwarm')
plt.show()

mat = np.random.randint(0,1000,(10,10))
plt.imshow(mat)
plt.colorbar()
plt.show()

df = pd.read_csv('/home/senthil/projects/tensorflow/Tensorflow-Bootcamp-master/00-Crash-Course-Basics/salaries.csv')
print(df)
df.plot(x='Salary', y='Age', kind='scatter')
plt.show()

