import tensorflow as tf
from keras import backend as K
from keras.layers import Dense
from keras.layers import Dropout
from keras.objectives import categorical_crossentropy
from tensorflow.examples.tutorials.mnist import input_data
from keras.metrics import categorical_accuracy as acc 

mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.Session()
K.set_session(sess)

image = tf.placeholder(dtype=tf.float32, shape=(None, 784))
labels = tf.placeholder(dtype=tf.float32, shape=(None, 10))

x = Dense(128, activation='relu')(image)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(10, activation='softmax')(x)

loss = tf.reduce_mean(categorical_crossentropy(predictions, labels))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
init_op = tf.global_variables_initializer()
sess.run(init_op)

with sess.as_default():
    for i in range(100):
        batch = mnist_data.next_batch(50)
        train_step.run(feed_dict=
            { image:batch[0], labels:batch[1], K.learning_phase(1)})

acc_value = acc(labels=labels, logits=predictions)
with sess.as_default():
    print(acc_value.eval(feed_dict=
        { image:mnist_data.test.images, labels:mnist_data.test.labels, K.learning_phase(0) }))

