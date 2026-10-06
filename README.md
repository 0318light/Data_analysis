# 📊 Data Analysis 실습 모음

주차별 데이터 분석 · 머신러닝 · 딥러닝 실습 코드를 정리한 저장소입니다.
모든 실습은 **Google Colab** 환경을 기준으로 작성되었습니다.

---

## 📁 폴더 구성

```
Data_analysis/
├── 4주차/
│   └── 4주차_assignment_2026 (1).ipynb      # pandas 기초 실습 (15문항)
├── 5주차/
│   └── My_First_Deeplearning.py             # 폐암 수술 환자 생존 예측 딥러닝
├── 6주차/
│   └── ShoppingMall_with_Clustering.ipynb   # K-Means · 계층적 군집 분석
└── README.md
```

| 주차 | 주제 | 파일 | 주요 라이브러리 |
|---|---|---|---|
| 4주차 | pandas 기초 (Series / DataFrame) | `4주차_assignment_2026 (1).ipynb` | pandas, seaborn |
| 5주차 | 첫 딥러닝 모델 (이진 분류) | `My_First_Deeplearning.py` | TensorFlow / Keras, NumPy |
| 6주차 | 비지도 학습 – 군집 분석 | `ShoppingMall_with_Clustering.ipynb` | scikit-learn, SciPy, yellowbrick, seaborn |

---

## 🐼 4주차: pandas 기초 실습

pandas의 **Series / DataFrame 핵심 문법을 익히는 15개의 실습 문제**로 구성되어 있습니다.

### 🛠️ 사용 라이브러리
* **pandas**
* **seaborn** (`titanic` 내장 데이터셋 로드용)
* **Google Colab** (`/content/sample_data/` 경로의 CSV 파일 사용)

### 📝 실습 내용

| 과제 | 내용 |
|---|---|
| 1 | 딕셔너리로 `Series` 생성 |
| 2 | 파이썬 리스트로 `Series` 생성 |
| 3 | 튜플 + `index` 옵션으로 `Series` 생성 |
| 4 | 딕셔너리로 `DataFrame` 생성 |
| 5 | 리스트 + 행/열 인덱스 지정으로 `DataFrame` 생성 |
| 6 | `rename()`으로 행 인덱스/열 이름 변경 (단계별 재지정 포함) |
| 7 | `drop()`으로 특정 행/열 삭제 (단일·복수, `axis` 옵션) |
| 8 | `loc`/`iloc`으로 행 선택 (단일·복수 행) |
| 9 | `loc`/`iloc`으로 특정 셀 및 다중 행·열 선택 |
| 10 | 열 추가, 행 추가(`loc`), 특정 값 수정 |
| 11 | `set_index()`로 인덱스 지정 후 `sort_index()`로 정렬 |
| 12 | `titanic` 데이터셋에 불리언 인덱싱 적용 (25세 이상 남성 필터링) |
| 13 | `query()`를 이용한 조건 필터링 (25세 이상 & pclass 3) *(선택)* |
| 14 | `Case.csv` 읽기 → 열 삭제 → 열 추가 → `case2.csv`로 저장 *(선택)* |
| 15 | `titanic.csv` 읽기 → 다중 열 삭제 → 열 이름 변경 → `titanic_new.csv`로 저장 *(선택)* |

### 🔑 핵심 학습 포인트
* `Series`/`DataFrame` 생성 방법 (딕셔너리, 리스트, 튜플 기반)
* 행/열 인덱스 및 이름 변경 (`rename`)
* 행/열 삭제 (`drop`)
* 위치 기반(`iloc`) vs 라벨 기반(`loc`) 인덱싱
* 데이터 추가 및 값 수정
* 인덱스 설정 및 정렬 (`set_index`, `sort_index`)
* 불리언 인덱싱과 `query()`를 활용한 조건 필터링
* CSV 파일 입출력 (`read_csv`, `to_csv`)

---

## 🫁 5주차: 폐암 수술 환자 생존 예측 딥러닝 (My First Deep Learning)

환자의 임상 기록 데이터를 바탕으로 수술 후 생존 여부를 예측하는 인공신경망 모델입니다.

### 🛠️ 사용 라이브러리
* **TensorFlow / Keras** (딥러닝 모델 구축)
* **NumPy** (데이터 로드 및 난수 고정)
* **Google Colab** (`google.colab.files`로 CSV 업로드)

### 📂 데이터셋
* **파일명**: `ThoraricSurgery.csv` (실행 시 직접 업로드)
* **입력 데이터 (X)**: 환자의 임상 기록 및 수술 관련 특징 17개 (인덱스 `0~16`)
* **결과 데이터 (Y)**: 수술 후 생존/사망 여부 (인덱스 `17`)

### 🧠 모델 구조
`Sequential` 모델로 층을 순서대로 쌓았습니다.

| 층 | 노드 수 | 입력 | 활성화 함수 |
|---|---|---|---|
| 은닉층 | 30 | 17개 특징 (`input_dim=17`) | `ReLU` |
| 출력층 | 1 | – | `Sigmoid` (0~1 확률 → 이진 분류) |

### ⚙️ 학습 설정
* **손실 함수**: `binary_crossentropy`
* **최적화 함수**: `adam`
* **평가 지표**: `accuracy`
* **하이퍼파라미터**: `epochs=100`, `batch_size=10`
* **난수 시드**: `np.random.seed(3)`, `tf.random.set_seed(3)` (재현성 확보)

---

## 🛍️ 6주차: 쇼핑몰 고객 군집 분석 (Clustering)

비지도 학습인 **K-Means**와 **계층적 군집(Hierarchical Clustering)** 을 간단한 예제로 익힌 뒤,
쇼핑몰 고객 데이터와 식품 영양 데이터에 적용해 보는 실습입니다.

### 🛠️ 사용 라이브러리
* **scikit-learn** (`KMeans`, `AgglomerativeClustering`, `StandardScaler`, `make_blobs`)
* **SciPy** (`linkage`, `dendrogram`)
* **yellowbrick** (`KElbowVisualizer` – 최적 K 탐색)
* **pandas / NumPy / matplotlib / seaborn**

### 📂 데이터셋
| 데이터 | 내용 |
|---|---|
| 직접 만든 배열 | 과일(신맛·단맛), 키·몸무게 등 2차원 소규모 예제 |
| `Mall_Customers.csv` | 쇼핑몰 고객 200명 – `Gender`, `Age`, `Annual Income (k$)`, `Spending Score (1-100)` |
| `food.csv` | 식품 영양 성분 – `carbohydrate`, `protein`, `fat`, `salt` |

### 📝 실습 흐름
1. **군집 기초** – 소규모 배열로 K-Means 학습, Elbow(inertia) 그래프, 덴드로그램, `AgglomerativeClustering`
2. **고객 데이터 EDA** – `head` / `info` / `describe`, 나이·연소득·소비점수 분포, 성별 비율, violin·swarm plot
3. **최적 K 탐색** – inertia 직접 계산 + `KElbowVisualizer`
4. **K-Means 군집화**
   * 나이 × 소비점수 → K=4
   * 연소득 × 소비점수 → K=5
   * 나이 × 연소득 × 소비점수 → K=5, `predict()`로 새 고객 군집 예측
5. **계층적 군집** – `ward` / `single` / `complete` / `average` 연결 방식 덴드로그램 비교, Agglomerative(K=5)
6. **응용 실습** – 키·몸무게 데이터, `food.csv` (결측치 제거 → `StandardScaler` → K-Means / 계층적 군집)

### 🔑 핵심 학습 포인트
* K-Means 원리와 `labels_`, `cluster_centers_`, `predict()` 활용
* Elbow 기법으로 적절한 군집 수(K) 선택
* 연결 방식(linkage)에 따른 계층적 군집 결과 차이
* 군집 결과를 DataFrame 라벨로 붙여 시각화·해석하기
* 스케일이 다른 변수는 표준화 후 군집화

---

## ▶️ 실행 방법

1. 원하는 주차의 `.ipynb` / `.py` 파일을 **Google Colab**에서 엽니다.
2. 필요한 CSV 파일을 Colab에 업로드합니다.
   * 4주차: `Case.csv`, `titanic.csv` → `/content/sample_data/`
   * 5주차: `ThoraricSurgery.csv` (실행 시 업로드 창)
   * 6주차: `Mall_Customers.csv`, `food.csv` → `/content/`
3. 6주차는 `yellowbrick`이 필요하면 `!pip install yellowbrick` 후 실행합니다.
