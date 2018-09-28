import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
mnist = input_data.read_data_sets("./data/", one_hot=True)
# 模型的输入和输出
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# 模型的权重和偏移量
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 创建Session
sess = tf.Session()
# 初始化权重变量
sess.run(tf.global_variables_initializer())

y = tf.nn.softmax(tf.matmul(x, W) + b)

# 交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# 训练
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 测试
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels}))
