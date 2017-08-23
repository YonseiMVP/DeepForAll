# Lab 3 Minimizing Cost
import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(777)  # for reproducibility
use_gpu = False
X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.placeholder(tf.float32)

# Our hypothesis for linear model X * W
hypothesis = X * W

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# GPU 사용 여부
if use_gpu == False:
    config = tf.ConfigProto(
    device_count={'GPU': 0} # GPU : 0이면 사용할 GPU 0개 -> CPU 사용
    )
elif use_gpu == True:
    config = tf.ConfigProto(
        device_count={'GPU': 1}  # GPU : 1이면 사용할 GPU 1개 -> GPU 사용
    )
# session에 그래프를 올린다. 그리고 그래프의 이름을 sess라고 정하겠다.
sess = tf.Session(config=config)

# Variables for plotting cost function
# histogram을 그리기 위하여 W, cost의 빈 리스트를 선언
W_history = []
cost_history = []

for i in range(-30, 50):
    curr_W = i * 0.1    #-3부터 5까지 0.1의 간격으로 값을 입력
    curr_cost = sess.run(cost, feed_dict={W: curr_W}) # 그래프 실행 -> W에 curr_W값을 입력으로 넣어준다.
    W_history.append(curr_W)    # W_history 리스트 끝에 curr_W값 추가
    cost_history.append(curr_cost)  #cost_history 리스트 끝에 curr_cost값 추가

# Show the cost function
plt.plot(W_history, cost_history)   #X-axis : W_history의 리스트 값, Y-axis : cost_history의 리스트 값
plt.show()  # cost function 그래프 plot
