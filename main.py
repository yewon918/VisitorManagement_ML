# https://wings2pc.tistory.com/entry/%EC%9B%B9-%EC%95%B1%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%B0%8D-%ED%8C%8C%EC%9D%B4%EC%8D%AC-%ED%94%8C%EB%9D%BC%EC%8A%A4%ED%81%ACPython-Flask?category=777829
# from flask import Flask, request
#
# app = Flask(__name__)    # flask를 app에 넘겨서 전역객체로 사용. 인스턴스 생성
# # 패키지 형태로 사용할 경우 패키지 이름을 __name__ 대신 직접 써줘야함. __name__은 단일모듈
#
# @app.route("/")
# def hello():
#     return 'hello'
#
# @app.route('/method', methods=['GET', 'POST'])
# def method():
#     if request.method == 'GET':
#         return "GET으로 전달"
#     else:
#         return "POST로 전달"
#
# if __name__ == '__main__':   # name은 main
#     app.run(debug=True)

#######################
import flask
from flask import Flask, request, render_template
import joblib
import numpy as np
from scipy import misc
import cv2


app = Flask(__name__)

# 메인페이지 - url 요청시 기본 index.html로 이동 (렌더링)
@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')

# 데이터
# 데이터 예측 처리
visit_list=[]
tmp = []
@app.route('/predict', methods=['POST', 'GET'])     # post인 것 같음
def make_prediction():
    if request.method == 'POST':

        # 업로드 파일 처리 분기  # 카메라로 이 부분을 받아오면 될거 같음

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('./trainer/trainer.yml')
        cascadePath = 'haarcascade_frontalface_default.xml'
        faceCascade = cv2.CascadeClassifier(
            r"C:\Users\Yewon\anaconda3\envs\py38\Library\etc\haarcascades\haarcascade_frontalface_default.xml")

        font = cv2.FONT_HERSHEY_SIMPLEX

        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1980)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        minW = 0.1 * cam.get(cv2.CAP_PROP_FRAME_WIDTH)
        minH = 0.1 * cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=6,
                minSize=(int(minW), int(minH))
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

                if confidence < 55:
                    put_name = id
                    # name을 거치지 않고 id를 출력하게 함. 이후 출력된 id는 db로 보내야함
                else:
                    put_name = -55

                confidence = "  {0}%".format(round(100 - confidence))

                cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
                cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

                tmp.append(id)

            cv2.imshow('camera', img)
            if cv2.waitKey(1) > 0: break

        visit_list.append(set(tmp))

        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()
        print(visit_list)

    return "완료"    # db로 보내줄 순 없을까?

if __name__ == '__main__':
    # 모델 로드
    # ml/model.py 선 실행 후 생성
    model = joblib.load('./model/face_recognition.pkl')
    # Flask 서비스 스타트
    app.run(host='0.0.0.0', port=8000, debug=True)
