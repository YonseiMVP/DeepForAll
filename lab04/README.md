
# 4. Multi-variable linear regression

이 코드들은 lab-02에서 진행한 단일 feature linear regression에서 더 나아가 multi variable feature에 대한 linear regression을 다루고 있습니다.

---
## 4.1 lab-04-1-multi_variable_linear_regression.py

4-1은 lab02에서 단일 feature에 대한 linear regression을 진행한 것에서 multi feature에 대한 linear regression을 진행하는 법을 소개합니다. 이 방법은 X, W, b, Y의 차원을 어떻게 설정해야하는 지에 대한 기본이 됩니다.

#### 실행 결과

<img src="img/lab-04-1.PNG">

## 4.2 lab-04-2-multi_variable_matmul_linear_regression.py

4-2은 4-1에서 hypothesis에 직접 x1,x2,x3,w1,w2,w3를 곱하였다면 간단하게 matmul이라는 함수를 이용하여 행렬 곱으로 multi variable linear regression을 진행하는 방법입니다.

#### 실행 결과

<img src="img/lab-04-2.PNG">

## 4.3 lab-04-3-file_input_linear_regression.py

4-3은 'data-01-test-score.csv'의 파일을 읽어 x_data(입력), y_data(label)로 분리하고 multi-variable linear regression을 진행하는 방법입니다. 여기서 x_data와 y_data를 어떤식으로 불러오는 지 중요합니다!

여기서 파일을 읽어오는 방식이 np.loadtxt인데 numpy를 써서 올리면 용량이 부족하게 됩니다.

#### 실행 결과

<img src="img/lab-04-3.PNG">

## 4.4 lab-04-4-tf_reader_linear_regression.py

4-4도 마찬가지로 파일을 불러오는 방법입니다. 4-3에서는 np.loadtxt로 읽었다면 이번에는 queue_runnere를 이용합니다. 사용 방법은 코드를 통해 이해할 수 있습니다.

#### 실행 결과

<img src="img/lab-04-4.PNG">
