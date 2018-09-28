import tensorflow as tf
hello=tf.constant('hello')
sess=tf.Session()
print(sess.run(hello))
print(sess.run(tf.cast(tf.greater(0.1,0.5),tf.int32)))