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

'''
when lr = 1.5
0 5.73203 [[-0.30548954  1.22985029 -0.66033536]
 [-4.39069986  2.29670858  2.99386835]
 [-3.34510708  2.09743214 -0.80419564]]
1 23.1494 [[ 0.06951046  0.29449689 -0.0999819 ]
 [-1.95319986 -1.63627958  4.48935604]
 [-0.90760708 -1.65020132  0.50593793]]
2 27.2798 [[ 0.44451016  0.85699677 -1.03748143]
 [ 0.48429942  0.98872018 -0.57314301]
 [ 1.52989244  1.16229868 -4.74406147]]
3 8.668 [[ 0.12396193  0.61504567 -0.47498202]
 [ 0.22003263 -0.2470119   0.9268558 ]
 [ 0.96035379  0.41933775 -3.43156195]]
4 5.77111 [[-0.9524312   1.13037777  0.08607888]
 [-3.78651619  2.26245379  2.42393875]
 [-3.07170963  3.14037919 -2.12054014]]
5 inf [[ nan  nan  nan]
 [ nan  nan  nan]
 [ nan  nan  nan]]
6 nan [[ nan  nan  nan]
 [ nan  nan  nan]
 [ nan  nan  nan]]

 ...
Prediction: [0 0 0]
Accuracy:  0.0

-------------------------------------------------
When lr = 1e-10
0 5.73203 [[ 0.80269563  0.67861295 -1.21728313]
 [-0.3051686  -0.3032113   1.50825703]
 [ 0.75722361 -0.7008909  -2.10820389]]
1 5.73203 [[ 0.80269563  0.67861295 -1.21728313]
 [-0.3051686  -0.3032113   1.50825703]
 [ 0.75722361 -0.7008909  -2.10820389]]
2 5.73203 [[ 0.80269563  0.67861295 -1.21728313]
 [-0.3051686  -0.3032113   1.50825703]
 [ 0.75722361 -0.7008909  -2.10820389]]
...

198 5.73203 [[ 0.80269563  0.67861295 -1.21728313]
 [-0.3051686  -0.3032113   1.50825703]
 [ 0.75722361 -0.7008909  -2.10820389]]
199 5.73203 [[ 0.80269563  0.67861295 -1.21728313]
 [-0.3051686  -0.3032113   1.50825703]
 [ 0.75722361 -0.7008909  -2.10820389]]
200 5.73203 [[ 0.80269563  0.67861295 -1.21728313]
 [-0.3051686  -0.3032113   1.50825703]
 [ 0.75722361 -0.7008909  -2.10820389]]
Prediction: [0 0 0]
Accuracy:  0.0
-------------------------------------------------
When lr = 0.1

0 5.73203 [[ 0.72881663  0.71536207 -1.18015325]
 [-0.57753736 -0.12988332  1.60729778]
 [ 0.48373488 -0.51433605 -2.02127004]]
1 3.318 [[ 0.66219079  0.74796319 -1.14612854]
 [-0.81948912  0.03000021  1.68936598]
 [ 0.23214608 -0.33772916 -1.94628811]]
2 2.0218 [[ 0.64342022  0.74127686 -1.12067163]
 [-0.81161296 -0.00900121  1.72049117]
 [ 0.2086665  -0.35079569 -1.909742  ]]

...

199 0.672261 [[-1.15377033  0.28146935  1.13632679]
 [ 0.37484586  0.18958236  0.33544877]
 [-0.35609841 -0.43973011 -1.25604188]]
200 0.670909 [[-1.15885413  0.28058422  1.14229572]
 [ 0.37609792  0.19073224  0.33304682]
 [-0.35536593 -0.44033223 -1.2561723 ]]
Prediction: [2 2 2]
Accuracy:  1.0
'''
