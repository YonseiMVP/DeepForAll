# Lab 4 Multi-variable linear regression
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility
use_gpu = False
# 3개의 feature , 5개의 instance, 차원은 instance x feature = 5 X 3
x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
# x_data와 동일한 방식
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]


# 입출력데이터를 넣기 위한 공간 (타입, 차원[None = instance 개수에 따라 자동으로 정해짐,feature 갯수를 맞춰주어야함]) => 나중에 feed_dict를 이용하여 값을 대입, trainable은 안됨
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# 변수선언(초기화 방법(차원),종류)노드 => trainable가능한
W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# hypothesis식을 정의 노드, tf.matmul은 텐서의 행렬곱 라이브러리 X=5x3, W=3x1
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
    # placeholder에 값을 넣어주고 cost, hypothesis 값을 cost_val, hy_val에 저장
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:  #10회당 cost, prediction값 출력)
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)
