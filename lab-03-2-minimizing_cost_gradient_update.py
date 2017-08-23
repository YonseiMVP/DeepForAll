# Lab 3 Minimizing Cost
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility
use_gpu = False
x_data = [1, 2, 3]
y_data = [1, 2, 3]

# Try to find values for W and b to compute y_data = W * x_data + b
# We know that W should be 1 and b should be 0
# But let's use TensorFlow to figure it out
W = tf.Variable(tf.random_normal([1]), name='weight')

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Our hypothesis for linear model X * W
hypothesis = X * W

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize: Gradient Descent using derivative: W -= learning_rate * derivative
learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)

# GPU 사용 여부
if use_gpu == False:
    config = tf.ConfigProto(
    device_count={'GPU': 0} # GPU : 0이면 사용할 GPU 0개 -> CPU 사용
    )
elif use_gpu == True:
    config = tf.ConfigProto(
        device_count={'GPU': 1}  # GPU : 1이면 사용할 GPU 1개 -> GPU 사용
    )
# session에 그래프를 올린다. 그리고 그래프의 이름을 sess라고 정하겠다.
sess = tf.Session(config=config)
# sess 그래프 안 변수들(W)에 지정한 random_normal distribution에 맞는 값으로 초기화
sess.run(tf.global_variables_initializer())

for step in range(21): # 21회 실행 0~20
    sess.run(update, feed_dict={X: x_data, Y: y_data})  #X placeholder에 x_data, Y placeholder에 y_data를 feed
    # 1. update -> W.assign(descent) : descent 값으로 W를 update한다.
    # 2. descent -> W - learning_rate * gradient : 그럼 descent 값은? W - learning_rate * gradient 이다.
    # 3. gradient -> tf.reduce_mean((W * X - Y) * X) : 그럼 gradient 값은?   (W * X - Y) * X 의 합의 평균 값
    # 4. X, Y 값은? feed_dict로 x_data, y_data가 들어 온다.
    # 5. 결론적으로 W값이 update 된다.

    print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))
    # step에 따른 cost, W 값의 변화를 출력

'''
0 1.93919 [ 1.64462376]
1 0.551591 [ 1.34379935]
2 0.156897 [ 1.18335962]
3 0.0446285 [ 1.09779179]
4 0.0126943 [ 1.05215561]
5 0.00361082 [ 1.0278163]
6 0.00102708 [ 1.01483536]
7 0.000292144 [ 1.00791216]
8 8.30968e-05 [ 1.00421977]
9 2.36361e-05 [ 1.00225055]
10 6.72385e-06 [ 1.00120032]
11 1.91239e-06 [ 1.00064015]
12 5.43968e-07 [ 1.00034142]
13 1.54591e-07 [ 1.00018203]
14 4.39416e-08 [ 1.00009704]
15 1.24913e-08 [ 1.00005174]
16 3.5322e-09 [ 1.00002754]
17 9.99824e-10 [ 1.00001466]
18 2.88878e-10 [ 1.00000787]
19 8.02487e-11 [ 1.00000417]
20 2.34053e-11 [ 1.00000226]
'''
