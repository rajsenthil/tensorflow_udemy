import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

x_data = np.linspace(0.0, 10.0, 1000000)
noise = np.random.randn(len(x_data))

# y = mx + b
# b = 5 and add some noise
y_true = (0.5 * x_data) + 5 + noise

x_df = pd.DataFrame(data=x_data, columns=['X Data'])
y_df = pd.DataFrame(data=y_true, columns=['Y'])

my_data = pd.concat([x_df, y_df], axis=1)
my_data.sample(n=250).plot(kind='scatter', x='X Data', y='Y')

plt.show()

batch_size = 8
rand = np.random.randn(2)
m = tf.Variable(rand[0])
b = tf.Variable(rand[1])


x_place_holder = tf.placeholder(tf.float64, [batch_size])
y_place_holder = tf.placeholder(tf.float64, [batch_size])

y_model = m * x_place_holder + b

error = tf.reduce_sum(tf.square(y_place_holder - y_model))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    batches = 1000

    for i in range(batches):
        rand_ind = np.random.randint(len(x_data), size=batch_size)
        feed = {x_place_holder: x_data[rand_ind], y_place_holder: y_true[rand_ind]}
        session.run(train, feed_dict=feed)

    model_m, model_b = session.run(([m,b]))

print(model_m)
print(model_b)

y_hat = x_data * model_m + model_b
my_data.sample(250).plot(kind='scatter', x='X Data', y='Y')
plt.plot(x_data, y_hat, 'r')
plt.show()