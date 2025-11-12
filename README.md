# 🚀 Deep Learning Course Repository

## 📘 소개
`gsc-lab/course-DL` 저장소는  
**영진전문대학교 글로벌시스템융합과 딥러닝 수업**을 위한 **예제 코드, 과제, 프로젝트 자료**를 제공합니다.  
학생들은 본 저장소를 클론하여 실습 환경을 구성하고 학습에 활용할 수 있습니다.

---

## 🛠️ 개발 환경 설정

### 1️⃣ 저장소 클론
```
git clone https://github.com/gsc-lab/course-DL.git
cd course-DL
```

### 2️⃣ 개발 환경 실행 (둘 중 하나 선택)

#### (방법 A) Docker CLI로 컨테이너 실행 → VS Code에서 **Attach**
1) 컨테이너 실행
```
docker compose up -d
```
2) VS Code에서 컨테이너에 붙기(Attach)
- 명령 팔레트 열기: **Ctrl/Cmd + Shift + P**
- **Dev Containers: Attach to Running Container...** 선택 후 `pytorch` 컨테이너 선택  
  (또는 Docker 확장 탭에서 컨테이너 우클릭 → **Attach Visual Studio Code**)
- VS Code가 컨테이너 내부의 `/workspace`를 워크스페이스로 열어줍니다.

> 참고: 방법 A는 컨테이너를 **수동으로 실행**한 뒤 VS Code가 **이미 실행 중인 컨테이너에 접속(Attach)** 하는 흐름입니다.

#### (방법 B, 추천) VS Code가 devcontainer 설정으로 **자동 생성 + Attach**
1) 프로젝트 폴더(`course-DL/`)를 VS Code로 열기
2) 우측 하단 팝업에서 **Reopen in Container** 클릭  
   → VS Code가 `devcontainer.json`과 `docker-compose.yml`의 설정을 읽어
   - 컨테이너를 **자동 생성/실행**하고
   - 워크스페이스를 **자동 Attach** 하며
   - 필요한 확장팩(**Python, Pylance, Jupyter**)을 **자동 설치**
   - `onCreateCommand` / `postCreateCommand`를 실행하여
     - 기본 유틸 설치(git)
     - 요구 패키지 설치(`requirements.txt`)
     - 커널 등록(`Python (PyTorch)`)
     - CUDA 사용 가능 여부 출력
   까지 자동으로 수행합니다.

> 방법 B를 사용하면 `docker compose up -d`를 **별도로 실행할 필요가 없습니다.**

### 3️⃣ Jupyter Notebook 사용
- `notebooks/` 폴더에서 `.ipynb` 파일 생성 또는 열기  
- 노트북 상단 메뉴에서 **Select Kernel → Python (PyTorch)** 선택 (devcontainer가 자동 등록)

테스트 코드:
```
import torch
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
```

---

## 📂 디렉토리 구조
```
course-DL/
 ├─ .devcontainer/       # VSCode Dev Container 설정 (devcontainer.json 등)
 ├─ data/                # 데이터셋 (raw/processed)
 ├─ notebooks/           # Jupyter 노트북 (실습/EDA)
 ├─ runs/                # 학습 결과 (로그, 체크포인트)
 ├─ src/                 # 학습/모델/유틸 코드
 ├─ tests/               # 단위 테스트 코드
 ├─ docker-compose.yml   # 컨테이너 실행 설정
 ├─ requirements.txt     # Python 패키지 목록
 └─ README.md
```

---

## ✅ 첫 실행 체크리스트
컨테이너 안에서 GPU 연결 여부를 확인하세요:
```
import torch
print("CUDA available:", torch.cuda.is_available())
```
출력이 **True**라면 GPU가 정상적으로 연결된 것입니다.
