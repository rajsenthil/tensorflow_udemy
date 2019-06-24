import tensorflow as tf

print(tf.__version__)

hello = tf.constant("Hello ")
world = tf.constant("World")

with tf.Session() as sess:
    result = sess.run(hello+world)

print(result)

a = tf.constant(10)
b = tf.constant(20)

with tf.Session() as sess:
    result = sess.run(a+b)

print(result)


const = tf.constant(10)

fill_mat = tf.fill((4,4), 10)
zeros = tf.zeros((4,4))
ones = tf.ones((4,4))

rand1 = tf.random_normal((4,4), mean=0, stddev=1.0)

rand2 = tf.random_normal((4,4), 0, 1)

ops = [const, fill_mat, zeros, ones, rand1, rand2]

sess = tf.InteractiveSession()

# for op in ops:
#     print(sess.run(op))
#     print('\n')

for op in ops:
    print(op.eval())
    print('\n')

a = tf.constant([[1,2],
                 [3,4]])

print(a.get_shape())

b = tf.constant([[10], [100]])

print(b.get_shape())

result = tf.matmul(a,b)

print(sess.run(result))