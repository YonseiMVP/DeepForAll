# Lab 5 Logistic Regression Classifier
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # for reproducibility
use_gpu = False
# numpy라이브러리를 이용하여 해당 경로 폴더안에 있는 ~.csv파일을 로드하는것
xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)

#로드한 xy행렬을 입력과 출력 데이터로 나누기 위한것 노드, -1이라는 것은 end-1로 생각하면 됨
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# ~.shape는 행렬의 차원을 보기위한
print(x_data.shape, y_data.shape)

# 입출력데이터를 넣기 위한 공간 (타입, 차원[None = instance 개수에 따라 자동으로 정해짐,feature 갯수를 맞춰주어야함]) => 나중에 feed_dict를 이용하여 값을 대입, trainable은 안됨
X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# 변수선언(초기화 방법(차원),종류)노드 => trainable가능한
W = tf.Variable(tf.random_normal([8, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# hypothesis식을 정의 노드+activation function(sigmoid)추가
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# cross entropy error 노드
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                       tf.log(1 - hypothesis))

# gradientdescent방법으로 초기화(학습속도 설정)하는 노드+gradientdescent방법으로 cost를 최소화하는 노드
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# cast는 if와 같은 역활 노드 0.5보다 크면 1 아니면 0
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# tf.equal은 두 인수가 같으면 1 아니면 0을 출력
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# GPU 사용 여부
if use_gpu == False:
    config = tf.ConfigProto(
    device_count={'GPU': 0} # GPU : 0이면 사용할 GPU 0개 -> CPU 사용
    )
elif use_gpu == True:
    config = tf.ConfigProto(
        device_count={'GPU': 1}  # GPU : 1이면 사용할 GPU 1개 -> GPU 사용
    )
with tf.Session(config=config) as sess:
    # sess 그래프 안 변수들(W, b)에 지정한 random_normal distribution에 맞는 값으로 초기화 된다.
    sess.run(tf.global_variables_initializer())

    for step in range(10001):   # 10001회 실행
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0: #200회마다 cost값 출력
            print(step, cost_val)

    # Accuracy report
    # 성능 평가
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)