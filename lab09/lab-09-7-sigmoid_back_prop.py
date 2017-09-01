"""
In this file, we will implement back propagations by hands

We will use the Sigmoid Cross Entropy loss function.
This is equivalent to tf.nn.sigmoid_softmax_with_logits(logits, labels)

[References]

1) Tensorflow Document (tf.nn.sigmoid_softmax_with_logits)
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

2) Neural Net Backprop in one slide! by Sung Kim
    https://docs.google.com/presentation/d/1_ZmtfEjLmhbuM_PqbDYMXXLAqeWN0HwuhcSKnUQZ6MM/edit#slide=id.g1ec1d04b5a_1_83

3) Back Propagation with Tensorflow by Dan Aloni
    http://blog.aloni.org/posts/backprop-with-tensorflow/

4) Yes you should understand backprop by Andrej Karpathy
    https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b#.cockptkn7


[Network Architecture]

Input: x
Layer1: x * W + b
Output layer = σ(Layer1)

Loss_i = - y * log(σ(Layer1)) - (1 - y) * log(1 - σ(Layer1))
Loss = tf.reduce_sum(Loss_i)

We want to compute that

dLoss/dW = ???
dLoss/db = ???

please read "Neural Net Backprop in one slide!" for deriving formulas

"""
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # for reproducibility
use_gpu = False

# .csv 데이터 로드
xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
# 입력 데이터
X_data = xy[:, 0:-1]
#N은 입력 데이터의 instance 개수
N = X_data.shape[0]
# 출력 데이터
y_data = xy[:, [-1]]

# y_data has labels from 0 ~ 6
print("y has one of the following values")
print(np.unique(y_data))

# x_data와 y_data 차원을 출력
print("Shape of X data: ", X_data.shape)
print("Shape of y data: ", y_data.shape)

nb_classes = 7  # 0 ~ 6

# 입출력데이터를 넣기 위한 공간
X = tf.placeholder(tf.float32, [None, 16])
y = tf.placeholder(tf.int32, [None, 1])  # 0 ~ 6

# y(label)값을 one_hot_code 형식으로 바꿈
target = tf.one_hot(y, nb_classes)  # one hot
target = tf.reshape(target, [-1, nb_classes])
target = tf.cast(target, tf.float32)

# 입출력데이터를 넣기 위한 공간
W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# sigma 함수 정의
def sigma(x):
    # sigmoid function
    # σ(x) = 1 / (1 + exp(-x))
    return 1. / (1. + tf.exp(-x))

# sigma 미분 함수 (for backpropagation)
def sigma_prime(x):
    # derivative of the sigmoid function
    # σ'(x) = σ(x) * (1 - σ(x))
    return sigma(x) * (1. - sigma(x))


# forward 과정
layer_1 = tf.matmul(X, W) + b
y_pred = sigma(layer_1)

# loss 계산
loss_i = - target * tf.log(y_pred) - (1. - target) * tf.log(1. - y_pred)
loss = tf.reduce_sum(loss_i)

# Dimension Check
assert y_pred.shape.as_list() == target.shape.as_list()


# backpropagation 과정 (chain rule을 이용함)(loss->sigma 순으로 chain rule)
d_loss = (y_pred - target) / (y_pred * (1. - y_pred) + 1e-7)
d_sigma = sigma_prime(layer_1)
d_layer = d_loss * d_sigma
d_b = d_layer
d_W = tf.matmul(tf.transpose(X), d_layer)

# gradient descent 이용하여 W값을 변경, assign을 이용하여 W에 새로운 값을 할당
learning_rate = 0.01
train_step = [
    tf.assign(W, W - learning_rate * d_W),
    tf.assign(b, b - learning_rate * tf.reduce_sum(d_b)),
]

# 정확도 계산
prediction = tf.argmax(y_pred, 1)
acct_mat = tf.equal(tf.argmax(y_pred, 1), tf.argmax(target, 1))
acct_res = tf.reduce_mean(tf.cast(acct_mat, tf.float32))

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

    for step in range(500): #training w9th backprop
        sess.run(train_step, feed_dict={X: X_data, y: y_data})

        if step % 10 == 0:
            # Within 300 steps, you should see an accuracy of 100%
            step_loss, acc = sess.run([loss, acct_res], feed_dict={
                                      X: X_data, y: y_data})
            print("Step: {:5}\t Loss: {:10.5f}\t Acc: {:.2%}" .format(
                step, step_loss, acc))

    # Let's see if we can predict
    # Prediction 결과 확인하기
    pred = sess.run(prediction, feed_dict={X: X_data})
    for p, y in zip(pred, y_data): # p, y의 인수들을 병렬로 순서대로 나열
        msg = "[{}]\t Prediction: {:d}\t True y: {:d}"
        print(msg.format(p == int(y[0]), p, int(y[0])))
