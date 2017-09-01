# -*- coding: utf-8 -*-
# Lab 2 Linear Regression
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility
use_gpu = False

# x_train, y_train 값 노드
x_train = [1, 2, 3]
y_train = [1, 2, 3]

# 변수선언(초기화 방법(차원),종류)노드 => trainable가능한
W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# hypothesis식을 정의 노드
hypothesis = x_train * W + b

# mean square error 노드
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# gradientdescent방법으로 초기화(학습속도 설정)하는 노드
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
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

# sess 그래프 안 변수들(W, b)에 지정한 random_normal distribution에 맞는 값으로 초기화
sess.run(tf.global_variables_initializer())

# Fit the line
# sess.run일 때에만 그래프가 실행, 그래프가 2000회 반복 실행된다.
for step in range(2001):
    sess.run(train)
    # 1. train 노드 -> optimizer.minimize(cost) 실행 -> cost 노드와 연결
    # 2. cost 노드 -> mean square (hypothesis - y_train) -> hypothesis 노드, y_train 노드와 연결
    # 3. hypothesis 노드 -> x_train 노드, W노드, b노드와 연결
    # 4. 학습은 cost가 minimize하는 방향으로 변수 W, b를 조절하게 된다.
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))
        #해당 step에서의 cost값, W값, b값을 출력

# Learns best fit W:[ 1.],  b:[ 0.]

'''
0 2.82329 [ 2.12867713] [-0.85235667]
20 0.190351 [ 1.53392804] [-1.05059612]
40 0.151357 [ 1.45725465] [-1.02391243]
...

1920 1.77484e-05 [ 1.00489295] [-0.01112291]
1940 1.61197e-05 [ 1.00466311] [-0.01060018]
1960 1.46397e-05 [ 1.004444] [-0.01010205]
1980 1.32962e-05 [ 1.00423515] [-0.00962736]
2000 1.20761e-05 [ 1.00403607] [-0.00917497]
'''
