# Lab 9 XOR-back_prop
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

# 입출력데이터를 넣기 위한 공간
W1 = tf.Variable(tf.random_normal([2, 2]), name='weight1')
b1 = tf.Variable(tf.random_normal([2]), name='bias1')
#l1식의 노드
l1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([2, 1]), name='weight2')
b2 = tf.Variable(tf.random_normal([1]), name='bias2')
#Y_pred 식의 정의 노드, l1이 입력
Y_pred = tf.sigmoid(tf.matmul(l1, W2) + b2)

# cost 계산 노드
cost = -tf.reduce_mean(Y * tf.log(Y_pred) + (1 - Y) *
                       tf.log(1 - Y_pred))

# Network
#          p1     a1           l1     p2     a2           l2 (y_pred)
# X -> (*) -> (+) -> (sigmoid) -> (*) -> (+) -> (sigmoid) -> (loss)
#       ^      ^                   ^      ^
#       |      |                   |      |
#       W1     b1                  W2     b2

# loss 미분 값
d_Y_pred = (Y_pred - Y) / (Y_pred * (1.0 - Y_pred) + 1e-7)

# backpropagation 과정 (loss->sigma 미분순)
d_sigma2 = Y_pred * (1 - Y_pred)
d_a2 = d_Y_pred * d_sigma2
d_p2 = d_a2
d_b2 = d_a2
d_W2 = tf.matmul(tf.transpose(l1), d_p2)


d_b2_mean = tf.reduce_mean(d_b2, axis=[0])
d_W2_mean = d_W2 / tf.cast(tf.shape(l1)[0], dtype=tf.float32)

# 위와 동일
d_l1 = tf.matmul(d_p2, tf.transpose(W2))
d_sigma1 = l1 * (1 - l1)
d_a1 = d_l1 * d_sigma1
d_b1 = d_a1
d_p1 = d_a1
d_W1 = tf.matmul(tf.transpose(X), d_a1)


d_W1_mean = d_W1 / tf.cast(tf.shape(X)[0], dtype=tf.float32)
d_b1_mean = tf.reduce_mean(d_b1, axis=[0])

# gradient descent 이용하여 W값을 변경, assign을 이용하여 W에 새로운 값을 할당
step = [
  tf.assign(W2, W2 - learning_rate * d_W2_mean),
  tf.assign(b2, b2 - learning_rate * d_b2_mean),
  tf.assign(W1, W1 - learning_rate * d_W1_mean),
  tf.assign(b1, b1 - learning_rate * d_b1_mean)
]

#정확도 계산
predicted = tf.cast(Y_pred > 0.5, dtype=tf.float32)
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

    print("shape", sess.run(tf.shape(X)[0], feed_dict={X: x_data})) #X의 shape 4 출력 : instance가 4개


    for i in range(10001):
        sess.run([step, cost], feed_dict={X: x_data, Y: y_data})    #training
        if i % 1000 == 0:
            print(i, sess.run([cost, d_W1], feed_dict={
                  X: x_data, Y: y_data}), sess.run([W1, W2]))

    # Accuracy report
    h, c, a = sess.run([Y_pred, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)