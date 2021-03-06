import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print(mnist.train.images)
print(mnist.train.num_examples)
print(mnist.test.num_examples)

single_image = mnist.train.images[1].reshape(28, 28)
# single_image = mnist.train.images[1]  # is a single dimensional numpy ndarray of 784
print(single_image.shape)
plt.imshow(single_image, cmap='gist_gray')
# plt.plot(single_image)
plt.show()

# Place holders
x = tf.placeholder(tf.float32, shape=[None, 784])

# Variables
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# Create Graph Operations
y = tf.matmul(x, W) + b

# Loss function
y_true = tf.placeholder(tf.float32, [None,10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))

# Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(cross_entropy)

# Create Session
init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)

    for step in range(1000):
        batch_x, batch_y = mnist.train.next_batch(100)
        session.run(train, feed_dict={x: batch_x, y_true: batch_y})

    # Evaluate the model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))

    # [True, False, True.....] => [1, 0, 1,.....]
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Predicted [3, 4] Tre [3, 9]
    # [True, False]
    # [1.0, 0.0]
    # 0.5

    print(session.run(accuracy, feed_dict={x: mnist.test.images, y_true: mnist.test.labels}))
