# coding=utf-8

from sklearn import svm, datasets, model_selection


if __name__ == '__main__':
    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.7)
    model = svm.SVC()
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test))
