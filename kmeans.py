import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

# 加载预处理后的数据
data_path = "/Users/wallenstein/Desktop/大二下/machine learning/Project/DSAA2011_Final_Project/DryBeanDataset/Processed_Dry_Beans.csv"
df = pd.read_csv(data_path)

# 提取数值特征（移除类别列）
category_columns = ['BARBUNYA', 'BOMBAY', 'CALI', 'DERMASON', 'HOROZ', 'SEKER', 'SIRA']
X = df.drop(columns=category_columns)

# 降维以便可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 使用 K-means 聚类
kmeans = KMeans(n_clusters=7, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

# 评估指标
kmeans_silhouette = silhouette_score(X, kmeans_labels)
kmeans_davies_bouldin = davies_bouldin_score(X, kmeans_labels)

# 打印评估结果
print("K-means Clustering:")
print(f"  Silhouette Score: {kmeans_silhouette:.2f}")
print(f"  Davies-Bouldin Score: {kmeans_davies_bouldin:.2f}")

# 可视化聚类结果
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', s=10)
plt.title("K-means Clustering")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster Label")
plt.tight_layout()
plt.savefig("/Users/wallenstein/Desktop/大二下/machine learning/Project/DSAA2011_Final_Project/DryBeanDataset/kmeans_clustering_results.png")
plt.show()