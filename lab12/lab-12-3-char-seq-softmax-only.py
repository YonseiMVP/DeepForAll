# Lab 12 Character Sequence Softmax only
import tensorflow as tf
import numpy as np
tf.set_random_seed(777)  # reproducibility
use_gpu = True
# 긴 문장에 대해 간편히 설정하기 위해
sample = " if you want you"
idx2char = list(set(sample))  # index -> char
char2idx = {c: i for i, c in enumerate(idx2char)}  # char -> idex

# hyper parameters
dic_size = len(char2idx)  # RNN input size (one hot size)
rnn_hidden_size = len(char2idx)  # RNN output size
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

# X를 num_class 개수에 따라 one_hot코드로 바꿔줌. No effect if the batch size is 1
X_one_hot = tf.one_hot(X, num_classes)  # one hot: 1 -> 0 1 0 0 0 0 0 0 0 0
# softmax를 사용하기 위해 reshape
X_for_softmax = tf.reshape(X_one_hot, [-1, rnn_hidden_size])

# softmax layer (rnn_hidden_size -> num_classes)
softmax_w = tf.get_variable("softmax_w", [rnn_hidden_size, num_classes])
softmax_b = tf.get_variable("softmax_b", [num_classes])
outputs = tf.matmul(X_for_softmax, softmax_w) + softmax_b

# sequence loss를 사용하기 위해 다시 원래대로 reshape
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])
# 각 output의 중요성,default는 동일하게=[1,1,1,...]
weights = tf.ones([batch_size, sequence_length])

# squence_loss(예측값,목표값, 각자리의 중요성)
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights)
# 최종 loss 계산
loss = tf.reduce_mean(sequence_loss)  # mean all sequence loss
# adamoptimizer방법으로 초기화(학습속도 설정)하는 노드 + adamoptimizer방법으로 cost를 최소화하는 노드
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

prediction = tf.argmax(outputs, axis=2)
if use_gpu == False:
    config = tf.ConfigProto(
    device_count={'GPU': 0} # uncomment this line to force CPU
    )
elif use_gpu == True:
    config = tf.ConfigProto(
        device_count={'GPU': 1}  # uncomment this line to force CPU
    )

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3000):
        l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_data})

        # print char using dic
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print(i, "loss:", l, "Prediction:", ''.join(result_str))

'''
0 loss: 2.29513 Prediction: yu yny y y oyny
1 loss: 2.10156 Prediction: yu ynu y y oynu
2 loss: 1.92344 Prediction: yu you y u  you

..

2997 loss: 0.277323 Prediction: yf you yant you
2998 loss: 0.277323 Prediction: yf you yant you
2999 loss: 0.277323 Prediction: yf you yant you
'''
