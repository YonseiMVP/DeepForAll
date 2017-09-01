# Lab 4 Multi-variable linear regression
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility
use_gpu = False
# x1,x2,x3은 각 feature에 해당, 5개는 instance
x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]
#출력값
y_data = [152., 185., 180., 196., 142.]

# 입출력데이터를 넣기 위한 공간 (타입, 차원[None = instance 개수에 따라 자동으로 정해짐]) => 나중에 feed_dict를 이용하여 값을 대입, trainable은 안됨
x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)

Y = tf.placeholder(tf.float32)

# 변수선언(초기화 방법(차원),종류)노드 => trainable가능한
w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
b = tf.Variable(tf.random_normal([1]), name='bias')

# hypothesis식을 정의 노드
hypothesis = x1 * w1 + x2 * w2 + x3 * w3 + b
print(hypothesis)

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
# session에 그래프를 올린다. 그리고 그래프의 이름을 sess라고 정함
sess = tf.Session(config=config)

# sess 그래프 안 변수들(w1,w2, w3, b)에 지정한 random_normal distribution에 맞는 값으로 초기화
sess.run(tf.global_variables_initializer())

for step in range(2001):
    # placeholder에 값을 넣어주고 cost, hypothesis 값을 cost_val, hy_val에 저장
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                   feed_dict={x1: x1_data, x2: x2_data, x3: x3_data, Y: y_data})

    if step % 10 == 0:  #10회당 cost, prediction값 출력)
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)