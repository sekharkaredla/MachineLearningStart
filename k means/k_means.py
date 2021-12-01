from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df_iris = pd.read_csv('IRIS.csv', header=None)
print(df_iris.info())

df_iris.describe()



sns.pairplot(df_iris)

data = df_iris.iloc[:, [0, 1, 2, 4]]
print(data)
data = data.values
print(data)

kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10)
kmeans.fit(data)


plt.show()