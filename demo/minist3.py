import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
import math
import matplotlib.image as mpimg
mnist = input_data.read_data_sets("./data/", one_hot=True)
# 模型的输入和输出
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# 创建Session
sess = tf.Session()


with tf.name_scope('hidden1') as scope:
    weights = tf.Variable(
        tf.truncated_normal([784, 10],
                            stddev=1.0 / math.sqrt(float(784))),name='weights')
    biases = tf.Variable(tf.zeros([10]),name='biases')

images=mpimg.imread('./data/person.png')
hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
logits = tf.matmul(hidden2, weights) + biases

batch_size = tf.size(50)
labels = tf.expand_dims(50, 1)
indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
concated = tf.concat(1, [indices, labels])
onehot_labels = tf.sparse_to_dense(concated, tf.pack([batch_size, 10]), 1.0, 0.0)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=onehot_labels,name='xentropy')
loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
tf.scalar_summary(loss.op.name, loss)
optimizer = tf.train.GradientDescentOptimizer(0.01)
global_step = tf.Variable(0, name='global_step', trainable=False)
train_op = optimizer.minimize(loss, global_step=global_step)
init = tf.initialize_all_variables()
sess.run(init)
for step in range(2000):
    sess.run(train_op)
images_feed, labels_feed =mnist.train.next_batch(100)

feed_dict = {
    x: images_feed,
    y_: labels_feed,
}
for step in range(2000):
    loss_value = sess.run([train_op, loss],feed_dict=feed_dict)
    if step % 100 == 0:
        print('Step %d: loss = %.2f ' % (step, loss_value))