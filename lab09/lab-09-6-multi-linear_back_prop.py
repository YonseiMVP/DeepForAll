# http://blog.aloni.org/posts/backprop-with-tensorflow/
# https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b#.b3rvzhx89
# WIP
import tensorflow as tf

tf.set_random_seed(777)  # reproducibility
use_gpu = False
# tf Graph Input
# 9-5 -> feature 1개 9-6 -> feature 3개(multi)
x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]

# 입출력데이터를 넣기 위한 공간
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# 변수선언(초기화 방법(차원),종류)노드 => trainable가능한
W = tf.Variable(tf.truncated_normal([3, 1]))
b = tf.Variable(5.)

# hypothesis식의 정의 노드 (forward 과정)
hypothesis = tf.matmul(X, W) + b

# shape을 통해 차원을 확인
print(hypothesis.shape, Y.shape)

# diff는 cost를 미분한 값
assert hypothesis.shape.as_list() == Y.shape.as_list()
diff = (hypothesis - Y)

# backpropagation 과정 (chain rule을 이용함), d_w는 cost를 w로 미분한 값
d_l1 = diff
d_b = d_l1
d_w = tf.matmul(tf.transpose(X), d_l1)

print(X, d_l1, d_w)

# gradient descent 이용하여 W값을 변경, assign을 이용하여 W에 새로운 값을 할당
learning_rate = 1e-6
step = [
    tf.assign(W, W - learning_rate * d_w),
    tf.assign(b, b - learning_rate * tf.reduce_mean(d_b)),
]

# mean square error 값을 계산
RMSE = tf.reduce_mean(tf.square((Y - hypothesis)))

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(10000):  # 9-5와 동일
    print(i, sess.run([step, RMSE], feed_dict={X: x_data, Y: y_data}))

print(sess.run(hypothesis, feed_dict={X: x_data}))
