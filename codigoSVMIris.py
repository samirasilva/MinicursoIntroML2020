from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm

iris = datasets.load_iris()

print(iris.data)
print(iris.target)
print(iris.data.shape)
print(iris.target.shape)

X_train, X_test, y_train, y_test = train_test_split (iris.data, iris.target, test_size=0.4, random_state=0)


clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
print(clf.score(X_test, y_test))


