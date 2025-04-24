# 📊 Performance_indicator

본 프로젝트는 `Main.py`의 `Performance` 클래스를 통해, **관 종별 Reference Marker의 매핑 변환 시 평균적인 정확도**를 평가합니다.

---

## 📌 주요 구성 요소

### 1. 🎯 성능 평가
- 각 관종에 대해 실제 Reference Marker (직경 19mm)의 매핑 변환 정확도를 **MAPE(%)** 기준으로 평가합니다.
- 매핑은 이미지에서 Reference Marker의 이진 이미지 영역을 변환한 뒤, 실제 면적과의 오차를 계산합니다.

### 2. 📷 카메라 파라미터
- 사용된 카메라의 **내부 파라미터** 및 **관 종별 직경 정보**는 [`Util/Camera_parameter.py`](./Util/Camera_parameter.py)에 포함되어 있습니다.
- **PVC 관**의 경우, **물을 넣어 촬영된 데이터셋**이 사용되었습니다.

### 3. 🧠 카메라 자세 회귀 모델
- 카메라 자세는 [`AI_based_CameraPose_Estimation.py`](./AI_based_CameraPose_Estimation.py)를 통해 예측됩니다.
- 예측 결과:
  - 이미지 내 소실점 좌표 위치 (x, y, % 단위)
  - 관로 중심 대비 x, y 위치 (비율 단위)
- 해당 출력은 `convision_output` 함수를 통해 이미지의 크기와 관 경에 맞게 변환됩니다.

### 4. 🗺️ 매핑 알고리즘
- 예측된 카메라 외부 파라미터를 활용해 [`Mapping_Algorithm.py`](./Mapping_Algorithm.py)에서 **평면 지도 형태로 매핑**이 수행됩니다.
- Reference Marker의 이진 이미지와 매핑된 결과를 비교하여 평균 정확도를 MAPE(%) 기준으로 산출합니다.

---

## 📥 다운로드
- 🔗 **[모델 다운로드](https://drive.google.com/file/d/1hQzI-rVq_Br3iA5lejnrL-r_7RyDUkZO/view?usp=drive_link)**
- 🔗 **[데이터셋 다운로드](https://drive.google.com/file/d/1ek2y7ZldS0WOKKWS6aWuvQdbuKTMbkLL/view?usp=drive_link)**
- 🔗 **[테스트에 사용된 가상환경(Conda) 다운로드](https://drive.google.com/file/d/1UdO8GioxjEhIxPFrlbxj2YyO_5R40N91/view?usp=drive_link)**

> 위 파일에는 학습된 모델 파일과 가상환경 설치에 필요한 `PPNet101.pt` 와 `Performance.yaml` 파일이 포함되어 있습니다.
