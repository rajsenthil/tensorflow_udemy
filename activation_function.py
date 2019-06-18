import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from operation import Operation
from operation import Graph
from operation import PlaceHolder
from operation import Variable
from operation import add
from operation import matmul
from operation import Session


def sigmoid(z):
    return 1/(1+np.exp(-z))


sample_z = np.linspace(-10, 10, 100)
sample_a = sigmoid(sample_z)

plt.plot(sample_z, sample_a)
plt.show()


class Sigmoid(Operation):
    def __init__(self, z):
        super().__init__([z])

    def compute(self, z_val):
        return 1/(1+np.exp(-z_val))


data = make_blobs(n_samples=50, n_features=2, centers=2, random_state=75)

features = data[0]
labels = data[1]

plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='coolwarm')
plt.show()

x = np.linspace(start=0, stop=11, num=10)
y = -x+5
plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='coolwarm')
plt.plot(x, y)
plt.show()

# d1 = np.array([1, 1]).dot(np.array([8], [10])) - 5

g = Graph()
g.set_as_default()
x = PlaceHolder()
w = Variable([1, 1])
b = Variable(-5)
z = add(matmul(w, x), b)
a = Sigmoid(z)
sess = Session()
print(sess.run(operation=a, feed_dict={x: [8, 10]}))
print(sess.run(operation=a, feed_dict={x: [2, -10]}))

