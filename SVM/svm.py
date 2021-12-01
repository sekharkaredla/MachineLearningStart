import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

iris = datasets.load_iris()
print(iris)
print(iris.values)

X = iris.data
Y = iris.target

print(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled)



fig = plt.figure(1, figsize=(16, 9))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = PCA(n_components=3).fit_transform(X_scaled)
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=Y,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

# plt.show()


X_train, X_test, y_train, y_test = train_test_split(
                        X_reduced, Y, test_size=0.2, random_state=42)

clf_SVC = SVC(C=100.0, kernel='rbf', gamma='auto', decision_function_shape="ovr", random_state = 0)
clf_SVC.fit(X_train,y_train)

print('Accuracy of SVC ovr on training set: {:.2f}'.format(clf_SVC.score(X_train, y_train) * 100))

print('Accuracy of SVC ovr on test set: {:.2f}'.format(clf_SVC.score(X_test, y_test) * 100))


clf_SVC = SVC(C=100.0, kernel='rbf', gamma='auto', decision_function_shape="ovo", random_state = 0)
clf_SVC.fit(X_train,y_train)

print('Accuracy of SVC ovo on training set: {:.2f}'.format(clf_SVC.score(X_train, y_train) * 100))

print('Accuracy of SVC ovo on test set: {:.2f}'.format(clf_SVC.score(X_test, y_test) * 100))

