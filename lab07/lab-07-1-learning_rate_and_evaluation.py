# Lab 7 Learning rate and Evaluation
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility
use_gpu = False
# 3개의 feature , 8개의 instance, 차원은 instance x feature = 8 X 3
x_data = [[1, 2, 1],
          [1, 3, 2],
          [1, 3, 4],
          [1, 5, 5],
          [1, 7, 5],
          [1, 2, 5],
          [1, 6, 6],
          [1, 7, 7]]
# 3개의 feature , 8개의 instance, 차원은 instance x feature = 8 X 3
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]


# test에 쓰기위한 데이터를 정의
x_test = [[2, 1, 1],
          [3, 1, 2],
          [3, 3, 4]]
y_test = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1]]

# 입출력데이터를 넣기 위한 공간
X = tf.placeholder("float", [None, 3])
Y = tf.placeholder("float", [None, 3])

# 변수선언(초기화 방법(차원),종류)노드 => trainable가능한
W = tf.Variable(tf.random_normal([3, 3]))
b = tf.Variable(tf.random_normal([3]))

# hypothesis식을 정의 노드
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# cross_entropy cost 값 게산 (axis=1은 괄호 가장 안의 값들을 더하는것)
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

# gradientdescent방법으로 초기화(학습속도 설정)하는 노드+gradientdescent방법으로 cost를 최소화하는 노드
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=1.5).minimize(cost) # <- 여기서 learning_rate를 수정하며 결과를 확인해보자! 1.5, 1e-10, 1e-1

# hypothesis의 각 행에서 가장 큰 인덱스 값을 출력 노드
prediction = tf.arg_max(hypothesis, 1)
# tf.equal은 두 인수가 같으면 1 아니면 0을 출력 노드
is_correct = tf.equal(prediction, tf.arg_max(Y, 1))
# 정확도 계산 노드
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

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
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    # 학습 시작
    for step in range(201):
        cost_val, W_val, _ = sess.run(
            [cost, W, optimizer], feed_dict={X: x_data, Y: y_data})
        print(step, cost_val, W_val)
    # 학습 끝

    # predict
    print("Prediction:", sess.run(prediction, feed_dict={X: x_test}))
    # Calculate the accuracy
    print("Accuracy: ", sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))