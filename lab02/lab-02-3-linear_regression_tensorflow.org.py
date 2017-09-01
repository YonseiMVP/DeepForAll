# From https://www.tensorflow.org/get_started/get_started
import tensorflow as tf
use_gpu = False
# 변수선언(초기화를 0.3 상수로 정해줌,종류)노드 => trainable가능한
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)

# 입출력데이터를 넣기 위한 공간 (타입, 차원[None = instance 개수에 따라 자동으로 정해짐]) => 나중에 feed_dict를 이용하여 값을 대입, trainable은 안됨
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# linear_model 식을 정의 노드
linear_model = x * W + b

# mean square error 노드
loss = tf.reduce_sum(tf.square(linear_model - y))  # sum of the squares

# gradientdescent방법으로 초기화(학습속도 설정)하는 노드
optimizer = tf.train.GradientDescentOptimizer(0.01)
# gradientdescent방법으로 cost를 최소화하는 노드
train = optimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# training loop
init = tf.global_variables_initializer()
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

# sess 그래프 안 변수들(W, b)에 지정한 값으로 초기화
sess.run(init)  # 잘못된 값을 넣어준다.
for i in range(1000):
    #x_train, y_train으로 train노드 ->  loss -> lineal_model, y 에 값이 들어간다.
    sess.run(train, {x: x_train, y: y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))
