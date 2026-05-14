import numpy as np
import joblib
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import logging
import datetime
import warnings
warnings.filterwarnings('ignore')
t = datetime.datetime.now()
t_now = t + datetime.timedelta(hours=9)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')
file_handler = logging.FileHandler('Logs.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

svm_model = joblib.load("/home/usr/svm.pkl")
testSet = np.load("/home/usr/testSet.npz")
X_test = testSet["X_test"]
y_test = testSet["y_test"]
testIds = testSet["ids"]
pred = svm_model.predict(X_test)

logger.debug("1) 시작 timestamp")
logger.debug(f"{t_now}\n")
logger.debug("\n2) 실행 명령어")
logger.debug("python svm.py")
logger.debug("\n3) 문항 개별 결과값")
logger.debug("데이터ID 모델예측값 GT값")
for i in range(len(X_test)):
	logger.debug(f"{testIds[i]} {pred[i]} {y_test[i]}")

logger.debug("\n4) 계산할 때 사용된 값")
logger.debug("클래스ID TP FP TN FN")
logger.debug("       0 75 31 52  8")
logger.debug("       1 52  8 75 31")

logger.debug("\n5) 최종 결과값\n")
logger.debug(f"{classification_report(y_test,pred,digits=4)}")

logger.debug("6) 종료 timestamp")
logger.debug(f"{t_now}")

