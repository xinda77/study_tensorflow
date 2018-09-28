#coding:utf-8
import  tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入数据集
mnist = input_data.read_data_sets("./data/", one_hot=True)
#每个批次的大小
batch_size = 100
#计算一共有多少个批次


n_batch = mnist.train.num_examples // batch_size

#参数概要 传入一个参数可以计算这个参数的各个相关值
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)#平均值
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)#标准差
        tf.summary.scalar('max', tf.reduce_max(var))#最大值
        tf.summary.scalar('min', tf.reduce_min(var))#最小值
        tf.summary.histogram('histogram', var)#直方图

with tf.name_scope('input'):
#定义两个placeholder
    x = tf.placeholder(tf.float32, [None,784],name='x-input') #输入图像
    y = tf.placeholder(tf.float32, [None,10],name='y-input') #输入标签
#创建一个简单的神经网络 784个像素点对应784个数  因此输入层是784个神经元 输出层是10个神经元 不含隐层
#最后准确率在92%左右
with tf.name_scope('layer'):
    with tf.name_scope('wights'):
        W = tf.Variable(tf.zeros([784,10]),name = 'W') #生成784行 10列的全0矩阵
        variable_summaries(W)
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([1,10]),name='b')
        variable_summaries(b)
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(tf.matmul(x,W)+b)

#二次代价函数
#loss = tf.reduce_mean(tf.square(y-prediction))
#交叉熵损失
with tf.name_scope('loss'):
    loss =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels =y,logits = prediction))
    tf.summary.scalar('loss',loss)
#使用梯度下降法
#train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss) #学习率一般设置比较小 收敛速度快

#初始化变量
init = tf.global_variables_initializer()

#结果存放在布尔型列表中
#argmax能给出某个tensor对象在某一维上的其数据最大值所在的索引值
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(prediction,1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        tf.summary.scalar('accuracy',accuracy)
#合并所有的summary
merged = tf.summary.merge_all()
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('D:\logs',sess.graph) #定义记录日志的位置
    for epoch in range(50):
        for batch in range(n_batch): #
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            summary,_ = sess.run([merged,train_step],feed_dict={x:batch_xs,y:batch_ys})
        writer.add_summary(summary,epoch) #将summary epoch 写入到writer
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images, y:mnist.test.labels})
        print ("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))