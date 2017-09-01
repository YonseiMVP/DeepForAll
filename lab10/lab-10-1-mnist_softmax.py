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

# GPU 사용 여부
if use_gpu == False:
    config = tf.ConfigProto(
        device_count={'GPU': 0} # GPU : 0이면 사용할 GPU 0개 -> CPU 사용
    )
elif use_gpu == True:
    config = tf.ConfigProto(
        device_count={'GPU': 1}  # GPU : 1이면 사용할 GPU 1개 -> GPU 사용
    )
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

# train my model
for epoch in range(training_epochs):    # 총 반복 15회
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):    # total_batch 만큼 반복 = mnist train num examples 모두 사용
        batch_xs, batch_ys = mnist.train.next_batch(batch_size) # batch size만큼 xs,ys에 할당
        feed_dict = {X: batch_xs, Y: batch_ys}  # 미리 feed_dict에 값을 선언한다.
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch # 1 epoch 당 cost

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

# Test model and check accuracy
# correct_prediction : True or False
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
# accuracy : tf.cast로 1, 0으로 반환 후 평균 구한다.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: mnist.test.images, Y: mnist.test.labels}))

# Get one and predict
# test example중 랜덤으로 한 example 실행
r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(
    tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))
