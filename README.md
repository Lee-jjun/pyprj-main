# PyPrj Django 프로젝트 설정 가이드

이 문서는 `pyprj` Django 프로젝트를 로컬 환경에 설정하고 실행하기 위한 기본 작업 순서를 정리한 가이드입니다.

---

## 1️⃣ Git 저장소 복제

먼저 GitHub 저장소를 로컬로 복제합니다.

```bash
git clone "https://github.com/nakrlove/pyprj.git"

가상환경에서 실행
pip install django

python manage.py makemigrations bbs
python manage.py migrate

---

### 🛠 유틸리티

#### Invisible space(U+00A0) 클리너 (git pull 후 실행)
가끔 파이썬 파일에 보이지 않는 특수 공백(U+00A0)이 들어가면  
`SyntaxError: invalid non-printable character U+00A0` 오류가 발생할 수 있습니다.  

이럴 땐 아래 유틸리티를 실행하면 전체 `.py` 파일을 검사해서 자동으로 수정해줍니다.

---

#### 사용 방법
1. 팀원이 `git pull` 하면 프로젝트 루트(`manage.py`가 있는 위치)에  
   **tools 폴더**와 그 안에 `clean_nbsp.py` 파일이 생성됩니다.

pyprj/
├── manage.py
├── requirements.txt
├── bbs/
├── jangoai/
└── tools/
└── clean_nbsp.py


2. 터미널에서 프로젝트 루트(`pyprj`) 위치에서 실행:
```bash
python tools/clean_nbsp.py

실행 결과

문제가 없으면 👌 no issue: ...

수정된 파일이 있으면 ✅ cleaned: ...

마지막에 요약(총 파일 수 / 수정된 파일 수 / 정상 파일 수)이 표시됩니다.