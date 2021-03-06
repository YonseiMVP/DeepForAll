# Lab 7 Learning rate and Evaluation
import tensorflow as tf
import random
import matplotlib.pyplot as mlp
# import matplotlib.pyplot as plt
tf.set_random_seed(777)  # for reproducibility
use_gpu = False

#tensorflow에서 mnist 다운을 받음
from tensorflow.examples.tutorials.mnist import input_data
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset
#mnist 데이터를 불러옴 (nmist의 label값을 one_hot코드로 불러옴)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_classes = 10

# mnist input 값을 넣기 위한 공간  28 * 28 = 784
X = tf.placeholder(tf.float32, [None, 784])
# Label값은 0 - 9 숫자이므로 총 10개의 class
Y = tf.placeholder(tf.float32, [None, nb_classes])

# 변수선언(초기화 방법(차원),종류)노드 => trainable가능한
W = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

# hypothesis식을 정의 노드
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# cross entropy error 노드
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

# gradientdescent방법으로 초기화(학습속도 설정)하는 노드+gradientdescent방법으로 cost를 최소화하는 노드
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# 출력값과 라벨값을 비교함, 같으면 1 다르면 0
is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
# 정확도 계산
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# parameters
training_epochs = 60
batch_size = 100

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
    # Training cycle
    for epoch in range(training_epochs): # 총 60회
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)    # train example 갯수 / batch_size만큼의 값

        for i in range(total_batch):    # total batch 만큼 반복
            batch_xs, batch_ys = mnist.train.next_batch(batch_size) #batch size만큼 batch_xs, ys에 불러온다.
            c, _ = sess.run([cost, optimizer], feed_dict={  # c에 cost 저장, optimizer은 값을 저장하지 않는다.
                            X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch # 1회 epoch의 cost 값을 avg_cost에 저장

        print('Epoch:', '%04d' % (epoch + 1),   #epoch 1당 평균 cost 값
              'cost =', '{:.9f}'.format(avg_cost))

    print("Learning finished")

    # Test the model using test sets
    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={
          X: mnist.test.images, Y: mnist.test.labels}))

    # Get one and predict
    #mnist test example 모든 샘플 중 random하게 골라 test
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("Prediction: ", sess.run(
        tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))
