
# 10. NN, ReLu, Xavier, Dropout, and Adam

�̹� �ǽ� 10���� MNIST�� �̿��� Neural Network ������ Ȯ���غ��� �ǽ��� �� �ֽ��ϴ�. ReLu, Xavier, Dropout �׸��� Adam optimizer�� ���� �߰��ذ��鼭 Accuracy�� ��� ���ϴ� �� Ȯ���ϰ� ������ �����ϸ� �����ϱ� �ٶ��ϴ�. 

10.1���� 11.2���� Training~Test ������ ���� �ּ��� 10.1�� 10.2���� ���� �������� ���� �ٶ��ϴ�.

---
## 10.1 lab-10-1-mnist_softmax.py

10-1�� MNIST�� ���� �ܼ��� ���� perceptron�� ���Ͽ� accuracy�� �˾ƺ��� �ڵ��Դϴ�. cost�Լ��� softmax�� ����� �� ���� function�� tf.nn.softmax_cross_entropy_with_logits�� ����Ͽ� �����Ͽ����ϴ�.

Test �� r���� ������ example number�� ���� �� ���� ���ϴ� ���ڸ� ������ ���Ƶ� �Ǹ� OpenCV Ȥ�� matplotlib�� ���� �̹����� ����Ͽ� Ȯ���غ� �� �ֽ��ϴ�.

#### ���� ���

<img src="img/lab-10-1.PNG">

## 10.2 lab-10-2-mnist_nn.py

10-2�� W1���� W3���� 3�� Neural Network�� �̿��Ͽ��� ��  accuracy�� ��� ���ϴ� �� Ȯ���� �� �ֽ��ϴ�.

#### ���� ���

<img src="img/lab-10-2.PNG">

## 10.3 lab-10-3-mnist_nn_xavier.py

10-3�� ó�� Weight�� bias���� initialize�� �� ����þ� ������ ���� ������ initializing�ϴ� ���� �ƴ� Xavier ������� initialize�Ͽ� ������ �� �� �̲��� �� �ڵ��Դϴ�. Xavier initializer�� ��� ���ǿ��� �� ����Ǿ� �ֽ��ϴ�.

#### ���� ���

<img src="img/lab-10-3.PNG">

## 10.4 lab-10-4-mnist_nn_deep.py

10-4�� ���� 5���� �÷��� �� ����� ��� ���ϴ� �� Ȯ���� �� �ֽ��ϴ�.

#### ���� ���

<img src="img/lab-10-4.PNG">

## 10.5 lab-10-5-mnist_nn_dropout.py

10-5�� dropout ��� (keep_prob�� ���� ����)�� �̿��Ͽ� ������ ��� ���ϴ� �� Ȯ���� �� �֤���ϴ�.

#### ���� ���

<img src="img/lab-10-5.PNG">

## 10.6 lab-10-6-mnist_nn_batchnorm.ipynb

10-6�� batch normalization�� ���� �ڵ��Դϴ�. ������ ��Ʈ������ �Ǿ� ������ �ڵ带 ���� �����ϰ� �м��� ������.