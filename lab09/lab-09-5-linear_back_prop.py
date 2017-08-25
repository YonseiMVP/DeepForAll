# http://blog.aloni.org/posts/backprop-with-tensorflow/
# https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b#.b3rvzhx89
# WIP
import tensorflow as tf

tf.set_random_seed(777)  # reproducibility
use_gpu = False

# tf Graph Input
x_data = [[1.],
          [2.],
          [3.]]
y_data = [[1.],
          [2.],
          [3.]]


# 입출력데이터를 넣기 위한 공간
X = tf.placeholder(tf.float32, shape=[None, 1])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# 변수선언(초기화 방법(차원),종류)노드 => trainable가능한, truncated normal distribution
W = tf.Variable(tf.truncated_normal([1, 1]))
b = tf.Variable(5.)

# hypothesis식의 정의 노드
hypothesis = tf.matmul(X, W) + b

# diff는 cost를 미분한 값
assert hypothesis.shape.as_list() == Y.shape.as_list()
diff = (hypothesis - Y)

# backpropagation 과정 (chain rule을 이용함), d_w는 cost를 w로 미분한 값
d_l1 = diff
d_b = d_l1
d_w = tf.matmul(tf.transpose(X), d_l1)

print(X, W, d_l1, d_w)

# gradient descent 이용하여 W값을 변경, assign을 이용하여 W에 새로운 값을 할당
learning_rate = 0.1
step = [
    tf.assign(W, W - learning_rate * d_w),
    tf.assign(b, b - learning_rate * tf.reduce_mean(d_b)),
]

# mean square error 값을 계산
RMSE = tf.reduce_mean(tf.square((Y - hypothesis)))

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    print(i, sess.run([step, RMSE], feed_dict={X: x_data, Y: y_data}))

print(sess.run(hypothesis, feed_dict={X: x_data}))
