# VisitorManagement_ML

---------

## 기본 설정

1) Anaconda 설치

2) 가상환경 만들기 (이름은 py38)

3) pip install -r requirements.txt


---------

## 파일 설명

ml: 머신러닝 코드가 들어가 있음

model: 훈련된 모델

static: 웹 css

templates: 웹 html

trainer: train시 필요한 yml 파일

main.py: 웹에서 카메라 작동, 판단

model.py: 모델 학습, 훈련

zolzak: 모델 초기 파일


---------

## 서버 실행 (Flask)

1) set FLASK_APP=main.py

2) flask run

3) http://127.0.0.1:5000/ 확인

> 개발서버 실행 시, set FLASK_DEBUG=true 이후 run flask
