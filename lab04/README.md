
# 4. Multi-variable linear regression

�� �ڵ���� lab-02���� ������ ���� feature linear regression���� �� ���ư� multi variable feature�� ���� linear regression�� �ٷ�� �ֽ��ϴ�.

---
## 4.1 lab-04-1-multi_variable_linear_regression.py

4-1�� lab02���� ���� feature�� ���� linear regression�� ������ �Ϳ��� multi feature�� ���� linear regression�� �����ϴ� ���� �Ұ��մϴ�. �� ����� X, W, b, Y�� ������ ��� �����ؾ��ϴ� ���� ���� �⺻�� �˴ϴ�.

#### ���� ���

<img src="img/lab-04-1.PNG">

## 4.2 lab-04-2-multi_variable_matmul_linear_regression.py

4-2�� 4-1���� hypothesis�� ���� x1,x2,x3,w1,w2,w3�� ���Ͽ��ٸ� �����ϰ� matmul�̶�� �Լ��� �̿��Ͽ� ��� ������ multi variable linear regression�� �����ϴ� ����Դϴ�.

#### ���� ���

<img src="img/lab-04-2.PNG">

## 4.3 lab-04-3-file_input_linear_regression.py

4-3�� 'data-01-test-score.csv'�� ������ �о� x_data(�Է�), y_data(label)�� �и��ϰ� multi-variable linear regression�� �����ϴ� ����Դϴ�. ���⼭ x_data�� y_data�� ������� �ҷ����� �� �߿��մϴ�!

���⼭ ������ �о���� ����� np.loadtxt�ε� numpy�� �Ἥ �ø��� �뷮�� �����ϰ� �˴ϴ�.

#### ���� ���

<img src="img/lab-04-3.PNG">

## 4.4 lab-04-4-tf_reader_linear_regression.py

4-4�� ���������� ������ �ҷ����� ����Դϴ�. 4-3������ np.loadtxt�� �о��ٸ� �̹����� queue_runnere�� �̿��մϴ�. ��� ����� �ڵ带 ���� ������ �� �ֽ��ϴ�.

#### ���� ���

<img src="img/lab-04-4.PNG">
