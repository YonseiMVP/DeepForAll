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
## tf.equal은 두 인수가 같으면 1 아니면 0을 출력 (True, False)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
# 정확도 계산 (tf.cast는 True, False를 1, 0으로 변환)
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
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})   # 그래프에 x_data, y_data를 feed 후 optimizer 실행
        if step % 100 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={
                                 X: x_data, Y: y_data})
            # 100회마다 loss에 cost값, acc에 accuracy 값을 입력
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(   # .format으로 {}마다 값을 넣어줄 수 있다.
                step, loss, acc))

    # Let's see if we can predict
    pred = sess.run(prediction, feed_dict={X: x_data})
    # y_data: (N,1) = flatten => (N, ) matches pred.shape
    for p, y in zip(pred, y_data.flatten()):    # zip은 여러개 list를 slice할 때 사용(김밥처럼)
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))