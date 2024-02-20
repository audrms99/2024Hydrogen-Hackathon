import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

class LOFModel:
    def __init__(self, n_neighbors=20, contamination=0.1):
        self.clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)

    def fit(self, X):
        self.clf.fit(X)

    def predict(self, X):
        return self.clf.fit_predict(X)

    def score_samples(self, X):
        return self.clf.negative_outlier_factor_

# CSV 파일에서 데이터 읽기
file_path = 'random_data.csv'
data_csv = pd.read_csv(file_path)

# 데이터 준비
X = data_csv.values

# LOF 모델 초기화 및 학습
lof_model = LOFModel(n_neighbors=20, contamination=0.1)
lof_model.fit(X)

# LOF 모델로 이상치 예측 및 점수 계산
y_pred = lof_model.predict(X)
scores = lof_model.score_samples(X)

# 이상치 시각화
# 시각화 코드를 추가하여 이상치를 적절히 표시할 수 있습니다.
import matplotlib.pyplot as plt

# 이상치 시각화
plt.figure(figsize=(10, 6))

# 이상치를 시각화할 때 사용할 색상
colors = np.array(['blue', 'red'])

# 이상치가 아닌 데이터 포인트를 먼저 플로팅합니다.
plt.scatter(X[:, 0], X[:, 1], c='blue', edgecolor='k', s=20)

# 이상치를 빨간색으로 플로팅합니다.
plt.scatter(X[y_pred == -1, 0], X[y_pred == -1, 1], c='red', edgecolor='k', s=80, marker='x', label='이상치')

plt.title('Determination of defective hydrogen')
plt.xlabel('temperature and humidity')
plt.ylabel('press')
plt.legend()
plt.grid(True)
plt.show()

