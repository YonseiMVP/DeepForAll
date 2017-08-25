# Lab 4 Multi-variable linear regression
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # for reproducibility
use_gpu = False

# numpy라이브러리를 이용하여 해당 경로 폴더안에 있는 ~.csv파일을 로드하는것
xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)

#로드한 xy행렬을 입력과 출력 데이터로 나누기 위한것 노드, -1이라는 것은 end-1로 생각하면 됨
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# ~.shape는 행렬의 차원을 보기위한 , 타입, 총 행렬 갯수)
print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)

# 입출력데이터를 넣기 위한 공간 (타입, 차원[None = instance 개수에 따라 자동으로 정해짐,feature 갯수를 맞춰주어야함]) => 나중에 feed_dict를 이용하여 값을 대입, trainable은 안됨
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# 변수선언(초기화 방법(차원),종류)노드 => trainable가능한
W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# hypothesis식을 정의 노드
hypothesis = tf.matmul(X, W) + b

# mean square error 노드
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# gradientdescent방법으로 초기화(학습속도 설정)하는 노드
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
# gradientdescent방법으로 cost를 최소화하는 노드
train = optimizer.minimize(cost)

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

# sess 그래프 안 변수들(W, b)에 지정한 random_normal distribution에 맞는 값으로 초기화 된다.
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

# Ask my score
print("Your score will be ", sess.run(
    hypothesis, feed_dict={X: [[100, 70, 101]]}))

print("Other scores will be ", sess.run(hypothesis,
                                        feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))

'''
Your score will be  [[ 181.73277283]]
Other scores will be  [[ 145.86265564]
 [ 187.23129272]]

'''
