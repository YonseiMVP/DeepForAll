# Lab 7 Learning rate and Evaluation
import tensorflow as tf
import random
#tensorflow에서 mnist 다운을 받고 label을 one_hot code방식으로 로드
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(777)  # reproducibility
use_gpu = False

#tensorflow에서 mnist 다운을 받음
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# parameters 설정
learning_rate = 0.001
training_epochs = 15
batch_size = 100

# mnist input 값을 넣기 위한 공간  28 * 28 = 784
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# 변수선언(초기화 방법(차원),종류)노드 => trainable가능한
W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.random_normal([10]))

# hypothesis식을 정의 노드
hypothesis = tf.matmul(X, W) + b

# cross entropy error 노드
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))

# adamoptimizer방법으로 초기화(학습속도 설정)하는 노드 + adamoptimizer방법으로 cost를 최소화하는 노드
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize.
if use_gpu == False:
    config = tf.ConfigProto(
    device_count={'GPU': 0} # uncomment this line to force CPU
    )
elif use_gpu == True:
    config = tf.ConfigProto(
        device_count={'GPU': 1}  # uncomment this line to force CPU
    )
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

# train my model
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: mnist.test.images, Y: mnist.test.labels}))

# Get one and predict
r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(
    tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

# plt.imshow(mnist.test.images[r:r + 1].
#           reshape(28, 28), cmap='Greys', interpolation='nearest')
# plt.show()

'''
Epoch: 0001 cost = 5.888845987
Epoch: 0002 cost = 1.860620173
Epoch: 0003 cost = 1.159035648
Epoch: 0004 cost = 0.892340870
Epoch: 0005 cost = 0.751155428
Epoch: 0006 cost = 0.662484806
Epoch: 0007 cost = 0.601544010
Epoch: 0008 cost = 0.556526115
Epoch: 0009 cost = 0.521186961
Epoch: 0010 cost = 0.493068354
Epoch: 0011 cost = 0.469686249
Epoch: 0012 cost = 0.449967254
Epoch: 0013 cost = 0.433519321
Epoch: 0014 cost = 0.419000337
Epoch: 0015 cost = 0.406490815
Learning Finished!
Accuracy: 0.9035
'''
