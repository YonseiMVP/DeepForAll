# Lab 9 XOR
import tensorflow as tf
import numpy as np

tf.set_random_seed(777)  # for reproducibility
learning_rate = 0.1
use_gpu = False

x_data = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]
y_data = [[0],
          [1],
          [1],
          [0]]
x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)

# 입출력데이터를 넣기 위한 공간
X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])

# 변수선언(초기화 방법(차원),종류)노드 => trainable가능한
W1 = tf.Variable(tf.random_normal([2, 10]), name='weight1')
b1 = tf.Variable(tf.random_normal([10]), name='bias1')
# layer1값의 식을 정의 노드
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([10, 10]), name='weight2')
b2 = tf.Variable(tf.random_normal([10]), name='bias2')\
# layer2값의 식을 정의 노드,layer1이 입력
layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

W3 = tf.Variable(tf.random_normal([10, 10]), name='weight3')
b3 = tf.Variable(tf.random_normal([10]), name='bias3')
# layer3값의 식을 정의 노드,layer2이 입력
layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)

W4 = tf.Variable(tf.random_normal([10, 1]), name='weight4')
b4 = tf.Variable(tf.random_normal([1]), name='bias4')
# hypothesos값의 식을 정의 노드,layer3이 입력
hypothesis = tf.sigmoid(tf.matmul(layer3, W4) + b4)

# cross entropy cost 노드
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                       tf.log(1 - hypothesis))
# gradientdescent방법으로 초기화(학습속도 설정)하는 노드+gradientdescent방법으로 cost를 최소화하는 노드
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# cast는 if문 hypothesis값이 0.5보다 크면 1 아니면 0
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# 정확도 계산
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# GPU 사용 여부
if use_gpu == False:
    config = tf.ConfigProto(
    device_count={'GPU': 0} # GPU : 0이면 사용할 GPU 0개 -> CPU 사용
    )
elif use_gpu == True:
    config = tf.ConfigProto(
        device_count={'GPU': 1}  # GPU : 1이면 사용할 GPU 1개 -> GPU 사용
    )
with tf.Session(config=config) as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(10001):   # 9-3 4층 구조
        sess.run(train, feed_dict={X: x_data, Y: y_data})   # train
        if step % 100 == 0:
            print(step, sess.run(cost, feed_dict={
                  X: x_data, Y: y_data}), sess.run([W1, W2]))

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)


'''
Hypothesis:  [[  7.80511764e-04]
 [  9.99238133e-01]
 [  9.98379230e-01]
 [  1.55659032e-03]]
Correct:  [[ 0.]
 [ 1.]
 [ 1.]
 [ 0.]]
Accuracy:  1.0
'''
