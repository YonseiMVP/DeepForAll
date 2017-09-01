# Lab 3 Minimizing Cost
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility
use_gpu = False

# x_data, y_data 값 노드
x_data = [1, 2, 3]
y_data = [1, 2, 3]

# 변수선언(초기화 방법(차원),종류)노드 => trainable가능한
W = tf.Variable(tf.random_normal([1]), name='weight')

# 입출력데이터를 넣기 위한 공간 (타입, 차원[None = instance 개수에 따라 자동으로 정해짐]) => 나중에 feed_dict를 이용하여 값을 대입, trainable은 안됨
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# hypothesis식을 정의 노드
hypothesis = X * W

# mean square error 노드
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# gradient 식을 직접 정의하는 코드 부분: W -= learning_rate * derivative
learning_rate = 0.1
# gradient 공식에 따라 설정
gradient = tf.reduce_mean((W * X - Y) * X)
# 새로운 weight 값을 update에 넣어준 후 assign 함수를 이용하여 weight에 다시 대입 W=W-learning_rate*gradient 와 같은 식으로는 오류남
descent = W - learning_rate * gradient
update = W.assign(descent)

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
# sess 그래프 안 변수들(W)에 지정한 random_normal distribution에 맞는 값으로 초기화
sess.run(tf.global_variables_initializer())

for step in range(21): # 21회 실행 0~20
    sess.run(update, feed_dict={X: x_data, Y: y_data})  #X placeholder에 x_data, Y placeholder에 y_data를 feed
    # 1. update -> W.assign(descent) : descent 값으로 W를 update한다.
    # 2. descent -> W - learning_rate * gradient : 그럼 descent 값은? W - learning_rate * gradient 이다.
    # 3. gradient -> tf.reduce_mean((W * X - Y) * X) : 그럼 gradient 값은?   (W * X - Y) * X 의 합의 평균 값
    # 4. X, Y 값은? feed_dict로 x_data, y_data가 들어 온다.
    # 5. 결론적으로 W값이 update 된다.

    print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))
    # step에 따른 cost, W 값의 변화를 출력