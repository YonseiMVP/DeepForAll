
# 12. RNN

12장에서는 CNN이 아닌 RNN에 대하여 다룰 것입니다. 보통 영상, 이미지 보다는 음성, 텍스트 등에 더 많이 사용된다고 하며 코드를 통해 RNN의 기초에 대해서 배워봅시다.

---
## 12.0 lab-12-0-rnn_basics.ipynb

12-0부터 이해하기 힘들 수 있습니다. 글자를 어떤 식으로 나누며 RNN의 데이터에 사용 되는 지 간단하게 살펴봅시다.
 
## 12.1 lab-12-1-hello_rnn.py

12-1은 CNN구조를 이용하여 MNIST 이미지를 flat한 상태가 아닌 28*28의 상태로 입력에 넣어 Training을 하는 코드 입니다.

#### 실행 결과

<img src="img/lab-12-1.PNG">

## 12.2 lab-12-2-char-seq-rnn.py

12-2는 

#### 실행 결과

<img src="img/lab-12-2.PNG">

## 12.3 lab-12-3-char-seq-softmax-only.py

12-3은 

#### 실행 결과

<img src="img/lab-12-3.PNG">

## 12.4 lab-12-4-rnn_long_char.py

12-4는 

#### 실행 결과

<img src="img/lab-12-4.PNG">

## 12.5 lab-12-5-rnn_stock_prediction.py

12-5은 강의에서 배운 앙상블 코드를 구현할 수 있습니다. 같은 구조의 네트워크들을 여러개 학습을 시켜 놓은 뒤 합치는 것으로 앙상블 코드를 구현할 수 있습니다.

#### 실행 결과



