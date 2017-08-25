# Lab 3 Minimizing Cost
# This is optional
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility
use_gpu = False
# X,Y 값 노드
X = [1, 2, 3]
Y = [1, 2, 3]

# 변수선언(상수로 초기화)노드 => trainable가능한
W = tf.Variable(5.)

# Linear model식을 정의 노드
hypothesis = X * W

# gradient 값을 추후 직접 출력하기 위한 노드
gradient = tf.reduce_mean((W * X - Y) * X) * 2

# # mean square error 노드
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# gradientdescent방법으로 초기화(학습속도 설정)하는 노드
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# gradientdescent방법으로 cost를 최소화하는 노드
train = optimizer.minimize(cost)

# cost의 gradient 값을 계산해줌 (customize한 gradient를 설정한다면)
gvs = optimizer.compute_gradients(cost, [W])
# gradient를 이용하여 어떻게 weight 수정을 할 것인지 설정
# Optional: modify gradient if necessary
# gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
# Apply gradients
apply_gradients = optimizer.apply_gradients(gvs)

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

for step in range(100):
    print(step, sess.run([gradient, W, gvs]))
    # gradient = 직접 구한 gradient
    # gvs = optimizer를 통하여 구한 gradient 값

    sess.run(apply_gradients)
    # Same as sess.run(train)
    # 결과가 같음을 볼 수 있다.

'''
# Apply gradients
0 [37.333332, 5.0, [(37.333336, 5.0)]]
1 [33.848888, 4.6266665, [(33.848888, 4.6266665)]]
2 [30.689657, 4.2881775, [(30.689657, 4.2881775)]]
3 [27.825287, 3.9812808, [(27.825287, 3.9812808)]]
4 [25.228262, 3.703028, [(25.228264, 3.703028)]]
...
96 [0.0030694802, 1.0003289, [(0.0030694804, 1.0003289)]]
97 [0.0027837753, 1.0002983, [(0.0027837753, 1.0002983)]]
98 [0.0025234222, 1.0002704, [(0.0025234222, 1.0002704)]]
99 [0.0022875469, 1.0002451, [(0.0022875469, 1.0002451)]]
'''
