import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

# 加载预处理后的数据
data_path = "/Users/wallenstein/Desktop/大二下/machine learning/Project/DSAA2011_Final_Project/DryBeanDataset/Processed_Dry_Beans.csv"
df = pd.read_csv(data_path)

# 提取数值特征（移除类别列）
category_columns = ['BARBUNYA', 'BOMBAY', 'CALI', 'DERMASON', 'HOROZ', 'SEKER', 'SIRA']
X = df.drop(columns=category_columns)

# 降维以便可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 使用层次聚类
hierarchical = AgglomerativeClustering(n_clusters=7, linkage='ward')
hierarchical_labels = hierarchical.fit_predict(X)

# 评估指标
hierarchical_silhouette = silhouette_score(X, hierarchical_labels)
hierarchical_davies_bouldin = davies_bouldin_score(X, hierarchical_labels)

# 打印评估结果
print("Hierarchical Clustering:")
print(f"  Silhouette Score: {hierarchical_silhouette:.2f}")
print(f"  Davies-Bouldin Score: {hierarchical_davies_bouldin:.2f}")

# 可视化聚类结果
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=hierarchical_labels, cmap='viridis', s=10)
plt.title("Hierarchical Clustering")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster Label")
plt.tight_layout()
plt.savefig("/Users/wallenstein/Desktop/大二下/machine learning/Project/DSAA2011_Final_Project/DryBeanDataset/hierarchical_clustering_results.png")
plt.show()

# 绘制层次聚类的树状图（Dendrogram）
plt.figure(figsize=(10, 7))
linked = linkage(X, method='ward')
dendrogram(linked, truncate_mode='level', p=5)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig("/Users/wallenstein/Desktop/大二下/machine learning/Project/DSAA2011_Final_Project/DryBeanDataset/hierarchical_dendrogram.png")
plt.show()