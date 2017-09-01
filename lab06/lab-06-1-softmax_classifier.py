# Lab 6 Softmax Classifier
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility
use_gpu = False
# 4개의 feature , 8개의 instance, 차원은 instance x feature = 8 X 4
x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
# x_data와 동일한 방식
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]
# 입출력데이터를 넣기 위한 공간
X = tf.placeholder("float", [None, 4])
Y = tf.placeholder("float", [None, 3])
nb_classes = 3

# 변수선언(초기화 방법(차원),종류)노드 => trainable가능한
W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# hypothesis식을 정의 노드 (softmax 함수를 사용)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# cross entropy error 노드
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

# gradientdescent방법으로 초기화(학습속도 설정)하는 노드+gradientdescent방법으로 cost를 최소화하는 노드
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

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
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0: # 200회마다 cost 값 출력
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

    print('--------------')

    # Testing & One-hot encoding
    a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9]]})
    print(a, sess.run(tf.arg_max(a, 1)))
    # 학습된 모델에 Test 데이터를 넣고 arg_max로 가장 큰 값의 index를 얻는다.
    print('--------------')

    b = sess.run(hypothesis, feed_dict={X: [[1, 3, 4, 3]]})
    print(b, sess.run(tf.arg_max(b, 1)))

    print('--------------')

    c = sess.run(hypothesis, feed_dict={X: [[1, 1, 0, 1]]})
    print(c, sess.run(tf.arg_max(c, 1)))

    print('--------------')

    #한번에 데이터를 입력하여 모두 볼 수 있다.
    all = sess.run(hypothesis, feed_dict={
                   X: [[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]})
    print(all, sess.run(tf.arg_max(all, 1)))

'''
--------------
[[  1.38904958e-03   9.98601854e-01   9.06129117e-06]] [1]
--------------
[[ 0.93119204  0.06290206  0.0059059 ]] [0]
--------------
[[  1.27327668e-08   3.34112905e-04   9.99665856e-01]] [2]
--------------
[[  1.38904958e-03   9.98601854e-01   9.06129117e-06]
 [  9.31192040e-01   6.29020557e-02   5.90589503e-03]
 [  1.27327668e-08   3.34112905e-04   9.99665856e-01]] [1 0 2]
'''
