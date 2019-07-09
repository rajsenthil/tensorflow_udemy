import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

n_features = 10
n_neurons = 3

x = tf.compat.v1.placeholder(tf.float32, (None, n_features))

W = tf.Variable(tf.random.normal([n_features, n_neurons]))

b = tf.Variable(tf.ones([n_neurons]))

xW = tf.matmul(x,W)

z = tf.add(xW, b)

a = tf.sigmoid(z)

init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as session:
    session.run(init)
    layer_out = session.run(a, feed_dict={x: np.random.random([1, n_features])})

print(layer_out)

x_data = np.linspace(0,10, 10) + np.random.uniform(-1.5, 1.5, 10)

# print(x_data)

y_label = np.linspace(0,10,10) + np.random.uniform(-1.5, 1.5, 10)

plt.plot(x_data, y_label, '*')
plt.show()


#y = mx + b
rand_2 = np.random.rand(2)
m = tf.compat.v1.Variable(rand_2[0], tf.float64)
b = tf.compat.v1.Variable(rand_2[1], tf.float64)
# rand_2 = tf.random(2)
# m = rand_2[0]
# b = rand_2[1]

error = tf.compat.v1.Variable(0, dtype=tf.float64)

for x,y in zip(x_data, y_label):
    y_hat = m*x + b

    error += (y-y_hat)**2

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

init = tf.compat.v1.global_variables_initializer()

saver = tf.train.Saver()

with tf.compat.v1.Session() as session:
    session.run(init)
    training_steps = 100

    for i in range(training_steps):
        session.run(train)

    final_slope, final_intercept = session.run(([m, b]))

    # Saving the trained model
    saver.save(session, 'models/tf_neural_network.ckpt')

with tf.compat.v1.Session() as session:
    # restoring the session
    saver.restore(session, 'models/tf_neural_network.ckpt')

    #Restoring the checkpoint
    restored_slope, restored_intervept = session.run([m,b])

x_test = np.linspace(-1, 11, 10)
y_pred_plot = final_slope * x_test + final_intercept
plt.plot(x_test, y_pred_plot, 'r')
plt.plot(x_data, y_label, '*')
plt.show()

