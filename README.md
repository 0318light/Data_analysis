# 🫁 3주차 Thoracic Surgery Deep Learning Model (폐암 수술 환자 생존 예측)

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

---

## 📊 4주차: Pandas 기초 실습 (Pandas Fundamentals)

4주차 과제(`4주차_assignment_2026.ipynb`)는 딥러닝 모델이 아닌, **pandas의 Series/DataFrame 핵심 문법을 익히는 15개의 실습 문제**로 구성되어 있습니다.

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
| 12 | `titanic` 데이터셋에 불리언 인덱싱(boolean indexing) 적용 (25세 이상 남성 필터링) |
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
