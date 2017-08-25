# Lab 3 Minimizing Cost
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility
use_gpu = False
# tf Graph Input
X = [1, 2, 3]
Y = [1, 2, 3]

# Set wrong model weights
W = tf.Variable(5.0)

# Linear model
hypothesis = X * W

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize: Gradient Descent Magic
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
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

# sess 그래프 안 변수들(W)에 지정한 random_normal distribution에 맞는 값으로 초기화 된다.
sess.run(tf.global_variables_initializer())

# 5.0이라는 값으로 잘못된 W(weight) 모델을 학습을 통해 변하는 것을 관찰
for step in range(100):
    print(step, sess.run(W))
    sess.run(train)
    # 1. train -> optimizer.minimize(cost) : GradientDescentOptimize을 이용 하여 cost를 minimize하는 방향으로 cost를 조정
    # 2. cost -> tf.reduce_mean(tf.square(hypothesis - Y)) : hypothesis - Y의 제곱의 합의 평균 을 최소화 하는 방향으로 hypothesis - Y(feed_dict)를 조정
    # 3. hypothesis -> X(feed_dict) * W에서 X * W - Y를 cost가 minimize하도록 값을 update, 즉 W 값을 train한다.

'''
0 5.0
1 1.26667
2 1.01778
3 1.00119
4 1.00008
...
96 1.0
97 1.0
98 1.0
99 1.0
'''