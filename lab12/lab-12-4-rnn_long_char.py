from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

tf.set_random_seed(777)  # reproducibility
use_gpu = True

sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

char_set = list(set(sentence))
char_dic = {w: i for i, w in enumerate(char_set)}

data_dim = len(char_set)
hidden_size = len(char_set)
num_classes = len(char_set)
sequence_length = 10  # Any arbitrary number
learning_rate = 0.1

dataX = []
dataY = []
for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i:i + sequence_length]
    y_str = sentence[i + 1: i + sequence_length + 1]
    print(i, x_str, '->', y_str)

    x = [char_dic[c] for c in x_str]  # x str to index
    y = [char_dic[c] for c in y_str]  # y str to index

    dataX.append(x)
    dataY.append(y)

batch_size = len(dataX)

# input값을 넣기 위한 공간 (문자열개수, 스펠링 개수)
X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])

# X를 num_class 개수에 따라 one_hot코드로 바꿔줌
X_one_hot = tf.one_hot(X, num_classes)
print(X_one_hot)  # check out the shape


# LSTM cell을 이용, 긴 코드이름을 cell 변수로 대체
def lstm_cell():
    cell = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
    return cell

# RNN 구조에서 여러개의 stack을 쌓는 구조             range(stack 쌓을 개수)
multi_cells = rnn.MultiRNNCell([lstm_cell() for _ in range(2)], state_is_tuple=True)

# 정의된 cell과 입력값을 rnn에 적용
outputs, _states = tf.nn.dynamic_rnn(multi_cells, X_one_hot, dtype=tf.float32)

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
mean_loss = tf.reduce_mean(sequence_loss)
# adamoptimizer방법으로 초기화(학습속도 설정)하는 노드 + adamoptimizer방법으로 cost를 최소화하는 노드
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mean_loss)
if use_gpu == False:
    config = tf.ConfigProto(
    device_count={'GPU': 0} # uncomment this line to force CPU
    )
elif use_gpu == True:
    config = tf.ConfigProto(
        device_count={'GPU': 1}  # uncomment this line to force CPU
    )

sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

for i in range(500):
    _, l, results = sess.run(
        [train_op, mean_loss, outputs], feed_dict={X: dataX, Y: dataY})
    for j, result in enumerate(results):
        index = np.argmax(result, axis=1)
        print(i, j, ''.join([char_set[t] for t in index]), l)

# Let's print the last char of each result to check it works
results = sess.run(outputs, feed_dict={X: dataX})
for j, result in enumerate(results):
    index = np.argmax(result, axis=1)
    if j is 0:  # print all for the first result to make a sentence
        print(''.join([char_set[t] for t in index]), end='')
    else:
        print(char_set[index[-1]], end='')

'''
0 167 tttttttttt 3.23111
0 168 tttttttttt 3.23111
0 169 tttttttttt 3.23111
…
499 167  of the se 0.229616
499 168 tf the sea 0.229616
499 169   the sea. 0.229616

g you want to build a ship, don't drum up people together to collect wood and don't assign them tasks and work, but rather teach them to long for the endless immensity of the sea.

'''
