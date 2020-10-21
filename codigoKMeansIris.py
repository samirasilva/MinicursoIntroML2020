#Código do K-Means
from sklearn import datasets
iris = datasets.load_iris()
X= iris.data
print(X)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3, init = 'random')

kmeans.fit(X)
print(kmeans.cluster_centers_)


distance = kmeans.fit_transform(X)
print(distance)

labels = kmeans.labels_
print(labels)


print(iris.target)

#Visualizando os dados do K-means 
import matplotlib.pyplot as plt
 
plt.scatter(X[:, 0], X[:,1], s = 100, c = kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'red',label = 'Centroides')
plt.title('Clusteres e Centroides do Dataset Iris')
plt.xlabel('Comprimento da Sépala')
plt.ylabel('Largura da Sépala')
plt.legend()
 
plt.show()
