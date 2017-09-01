
# 3. Minimizing Cost

이 코드들은 TensorFlow로 linear regression의 cost를 최소화하는 방법을 구현하였습니다.

---
## 3.1 lab-03-1-minimizing_cost_show_graph.py

3-1은 cost function을 최소화하는 방법과 그래프를 관찰할 수 있는 코드입니다. TensorFlow에서 어떤 함수를 이용하는 지 살펴봅시다.

#### 실행 결과

<img src="img/lab-03-1.PNG">

## 3.2 lab-03-2-minimizing_cost_gradient_update.py

3-2는 직접 gradient descent를 구현할 수 있는 코드입니다. assign을 통해 Weight 값을 새로 업데이트 하며 cost function을 최소화 합니다.

#### 실행 결과

<img src="img/lab-03-2.PNG">

## 3.3 lab-03-3-minimizing_cost_tf_optimizer.py

3-3은 optimizer을 이용하여 gradient descent를 사용하며 cost를 최소화하는 코드입니다.

#### 실행 결과

<img src="img/lab-03-3-1.PNG">

<img src="img/lab-03-3-2.PNG">

## 3.4 lab-03-X-minimizing_cost_tf_gradient.py

3-4는 직접 구한 gradient와 optimizer을 통해 구한 gradient의 값을 서로 비교해 볼 수 있는 코드입니다.

#### 실행 결과

<img src="img/lab-03-X-1.PNG">

<img src="img/lab-03-X-2.PNG">

