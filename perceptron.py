import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


training_inputs = np.array([[0, 0, 1],
                            [1, 1, 1],
                            [1, 0, 1],
                            [0, 1, 1]])

training_outputs = np.array([[0, 1, 1, 0]]).T

np.random.seed(1)

synaptics_weights = 2 * np.random.random((3, 1)) - 1

print('Random starting synaptics weights: ')
print(synaptics_weights)

for iteration in range(100000):

    input_layer = training_inputs

    outputs = sigmoid(np.dot(input_layer, synaptics_weights))

    error = training_outputs - outputs

    adjustments = error * sigmoid_derivative(outputs)

    synaptics_weights += np.dot(input_layer.T, adjustments)

print("Synaptic weights after training")
print(synaptics_weights)

print('Outouts after training: ')
print(outputs)


sample_z = np.linspace(-6, 6, 100)
sample_a = sigmoid(sample_z)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# Move left y-axis and bottim x-axis to centre, passing through (0,0)
ax.spines['left'].set_position('center')
#ax.spines['bottom'].set_position('center')

# Eliminate upper and right axes
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Show ticks in the left and lower axes only
#ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.plot(sample_z, sample_a)
plt.show()
