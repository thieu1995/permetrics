#!/usr/bin/env python
# Created by "Thieu" at 23:24, 28/06/2026 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import time
import permetrics.utils.cluster_util as cut
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
import hdbscan
from kDBCV import DBCV_score

np.random.seed(100)

# 1. Tạo dữ liệu hình bán nguyệt (arbitrary shape)
X, y = make_moons(n_samples=500, noise=0.1, random_state=42)

# 2. Tiến hành phân cụm bằng DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=5)
labels = dbscan.fit_predict(X)

# 3. Tính toán DBCV score thông qua hàm validity_index của hdbscan
# Lưu ý: Hàm này yêu cầu ma trận khoảng cách (prediction_matrix)
t1 = time.perf_counter()
dbcv_score = hdbscan.validity.validity_index(X.astype(np.float64), labels)
t1_end = time.perf_counter() - t1
print(f"DBCV Score của kết quả phân cụm: {dbcv_score:.4f}, time: {t1_end}")

t2 = time.perf_counter()
r1, r2 = cut.calculate_dbcv_score(X=X, y_pred=labels)
t2_end = time.perf_counter() - t2
print(f"My res: {r1}, time: {t2_end}")
print(r2)


# Tính toán score trực tiếp từ data và labels
# ind_clust_scores=True nếu bạn muốn xem điểm số của từng cụm riêng biệt
t3 = time.perf_counter()
score, ind_clust_scores = DBCV_score(X, labels, ind_clust_scores=True)
t3_end = time.perf_counter() - t3
print(f"Tổng điểm DBCV toàn cục: {score:.4f}, time: {t3_end}")
print(f"Điểm DBCV của từng cụm: {ind_clust_scores}")



# 4. Trực quan hóa dữ liệu
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50)
plt.title(f"DBSCAN Clustering (DBCV Score: {dbcv_score:.4f})")
plt.show()

