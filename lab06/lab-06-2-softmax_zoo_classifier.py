# Lab 6 Softmax Classifier
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # for reproducibility
use_gpu = False
# numpy라이브러리를 이용하여 해당 경로 폴더안에 있는 ~.csv파일을 로드하는것, 나머지 인수들은 default라고 생각하면됨
xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
#로드한 xy행렬을 입력과 출력 데이터로 나누기 위한것 노드, -1이라는 것은 end-1로 생각하면 됨
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
# ~.shape는 행렬의 차원을 보기위한
print(x_data.shape, y_data.shape)

nb_classes = 7  # 0 ~ 6

# 입출력데이터를 넣기 위한 공간
X = tf.placeholder(tf.float32, [None, 16])
Y = tf.placeholder(tf.int32, [None, 1])  # 0 ~ 6

# Y의 scalar 값을 binary를 가진 one_hot 벡터로 만들어준다.
Y_one_hot = tf.one_hot(Y, nb_classes)  # one hot
print("one_hot", Y_one_hot)
# one_hot은 행렬 차원을 하나더 늘리기 때문에 reshape로 다시 원하는 모양(2차원)으로 만들어 주어야함
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
print("reshape", Y_one_hot)

# 변수선언(초기화 방법(차원),종류)노드 => trainable가능한
W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# logits식을 정의 노드
logits = tf.matmul(X, W) + b
#hypothesis식을 정의 노드 (softmax 사용)
hypothesis = tf.nn.softmax(logits)

# logits식에 softmax와 cross_entropy 를 한번에 사용하게 해주는 함수 logits=인풋값, label=one_hot coding된 label값
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                 labels=Y_one_hot)
# cost계산
cost = tf.reduce_mean(cost_i)

# gradientdescent방법으로 초기화(학습속도 설정)하는 노드+gradientdescent방법으로 cost를 최소화하는 노드
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# hypothesis의 각 행에서 가장 큰 인덱스 값을 출력
prediction = tf.argmax(hypothesis, 1)
## tf.equal은 두 인수가 같으면 1 아니면 0을 출력
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
# 정확도 계산
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
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

    for step in range(2000):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={
                                 X: x_data, Y: y_data})
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(
                step, loss, acc))

    # Let's see if we can predict
    pred = sess.run(prediction, feed_dict={X: x_data})
    # y_data: (N,1) = flatten => (N, ) matches pred.shape
    for p, y in zip(pred, y_data.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))

'''
Step:     0 Loss: 5.106 Acc: 37.62%
Step:   100 Loss: 0.800 Acc: 79.21%
Step:   200 Loss: 0.486 Acc: 88.12%
Step:   300 Loss: 0.349 Acc: 90.10%
Step:   400 Loss: 0.272 Acc: 94.06%
Step:   500 Loss: 0.222 Acc: 95.05%
Step:   600 Loss: 0.187 Acc: 97.03%
Step:   700 Loss: 0.161 Acc: 97.03%
Step:   800 Loss: 0.140 Acc: 97.03%
Step:   900 Loss: 0.124 Acc: 97.03%
Step:  1000 Loss: 0.111 Acc: 97.03%
Step:  1100 Loss: 0.101 Acc: 99.01%
Step:  1200 Loss: 0.092 Acc: 100.00%
Step:  1300 Loss: 0.084 Acc: 100.00%
...
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 3 True Y: 3
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 3 True Y: 3
[True] Prediction: 3 True Y: 3
[True] Prediction: 0 True Y: 0
'''
