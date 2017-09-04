# Lab 12 Character Sequence RNN
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # reproducibility
use_gpu = True


sample = " if you want you"
idx2char = list(set(sample))  # index -> char
char2idx = {c: i for i, c in enumerate(idx2char)}  # char -> idex

# hyper parameters
dic_size = len(char2idx)  # RNN input size (one hot size)
hidden_size = len(char2idx)  # RNN output size
num_classes = len(char2idx)  # final output size (RNN or softmax, etc.)
batch_size = 1  # one sample data, one batch
sequence_length = len(sample) - 1  # number of lstm rollings (unit #)
learning_rate = 0.1

sample_idx = [char2idx[c] for c in sample]  # char to index
x_data = [sample_idx[:-1]]  # X data sample (0 ~ n-1) hello: hell
y_data = [sample_idx[1:]]   # Y label sample (1 ~ n) hello: ello

# input값을 넣기 위한 공간 (문자열개수, 스펠링 개수)
X = tf.placeholder(tf.int32, [None, sequence_length])  # X data
Y = tf.placeholder(tf.int32, [None, sequence_length])  # Y label

# X를 num_class 개수에 따라 one_hot코드로 바꿔줌
x_one_hot = tf.one_hot(X, num_classes)  # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0
# cell을 정의 hidden_size는 ouptsize
cell = tf.contrib.rnn.BasicLSTMCell(
    num_units=hidden_size, state_is_tuple=True)
# cell초기화
initial_state = cell.zero_state(batch_size, tf.float32)
# 정의된 cell과 입력값을 rnn에 적용
outputs, _states = tf.nn.dynamic_rnn(
    cell, x_one_hot, initial_state=initial_state, dtype=tf.float32)

# 각 output 결과를 일렬로 나열 FC layer
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
# fc_w = tf.get_variable("fc_w", [hidden_size, num_classes])
# fc_b = tf.get_variable("fc_b", [num_classes])
# outputs = tf.matmul(X_for_fc, fc_w) + fc_b
outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)

# sequence loss를 사용하기 위해 다시 원래대로 reshape
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])
# 각 output의 중요성,default는 동일하게=[1,1,1,...]
weights = tf.ones([batch_size, sequence_length])
# squence_loss(예측값,목표값, 각자리의 중요성)
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights)
# 최종 loss 계산
loss = tf.reduce_mean(sequence_loss)
# adamoptimizer방법으로 초기화(학습속도 설정)하는 노드 + adamoptimizer방법으로 cost를 최소화하는 노드
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)
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
    for i in range(50):
        l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_data})

        # print char using dic
        result_str = [idx2char[c] for c in np.squeeze(result)]

        print(i, "loss:", l, "Prediction:", ''.join(result_str))


'''
0 loss: 2.35377 Prediction: uuuuuuuuuuuuuuu
1 loss: 2.21383 Prediction: yy you y    you
2 loss: 2.04317 Prediction: yy yoo       ou
3 loss: 1.85869 Prediction: yy  ou      uou
4 loss: 1.65096 Prediction: yy you  a   you
5 loss: 1.40243 Prediction: yy you yan  you
6 loss: 1.12986 Prediction: yy you wann you
7 loss: 0.907699 Prediction: yy you want you
8 loss: 0.687401 Prediction: yf you want you
9 loss: 0.508868 Prediction: yf you want you
10 loss: 0.379423 Prediction: yf you want you
11 loss: 0.282956 Prediction: if you want you
12 loss: 0.208561 Prediction: if you want you

...

'''
