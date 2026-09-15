이 코드는 텐서플로(TensorFlow)와 케라스(Keras)를 이용해 흉부외과 수술 환자의 생존 여부를 예측하는 **이진 분류(Binary Classification) 딥러닝 모델**을 구현한 예제입니다.

GitHub 등의 프로젝트 **README.md** 파일에 바로 복사해서 붙여넣기 좋은 마크다운 형식으로 정리해 드립니다.

---

# 🫁 Thoracic Surgery Deep Learning Model (폐암 수술 환자 생존 예측)

이 프로젝트는 환자의 임상 기록 데이터를 바탕으로 수술 후 생존 여부를 예측하는 인공신경망 딥러닝 모델입니다. 파이썬의 `TensorFlow`와 `Keras` 라이브러리를 활용하여 구현되었습니다.

---

## 🛠️ 개발 환경 및 라이브러리

* **Python**
* **TensorFlow / Keras** (딥러닝 모델 구축)
* **NumPy** (데이터 처리 및 난수 고정)
* **Google Colab** (`google.colab.files`를 통한 파일 업로드 지원)

---

## 📂 데이터셋 (Dataset)

* **파일명**: `ThoraricSurgery.csv`
* **입력 데이터 ($X$)**: 환자의 임상 기록 및 수술 관련 특징 (총 17개 변수, 인덱스 `0~16`)
* **결과 데이터 ($Y$)**: 수술 후 생존/사망 여부 (타겟 변수, 인덱스 `17`)

---

## 🧠 모델 구조 (Model Architecture)

본 모델은 입력층부터 출력층까지 순차적으로 층을 쌓는 `Sequential` 모델을 사용했습니다.

* **입력층 / 은닉층 (Hidden Layer)**:
* 노드(뉴런) 수: `30개`
* 입력 변수 개수 (`input_dim`): `17개`
* 활성화 함수 (`activation`): `ReLU` (비선형성 부여)


* **출력층 (Output Layer)**:
* 노드 수: `1개` (생존 여부 확률 출력)
* 활성화 함수 (`activation`): `Sigmoid` (0과 1 사이의 확률값으로 변환하여 이진 분류 수행)

---

## ⚙️ 학습 설정 (Compilation & Training)

* **손실 함수 (Loss Function)**: `binary_crossentropy` (이진 분류 문제에 최적화된 교차 엔트로피 사용)
* **최적화 함수 (Optimizer)**: `adam` (효율적인 경사 하강법 알고리즘)
* **평가 지표 (Metrics)**: `accuracy` (정확도)
* **하이퍼파라미터**:
* 에포크 (`epochs`): `100` (전체 데이터셋 학습 반복 횟수)
* 배치 크기 (`batch_size`): `10` (한 번에 학습하는 데이터 묶음 크기)
* 난수 시드 (`seed`): `3` (실행할 때마다 동일한 결과를 얻기 위해 고정)
