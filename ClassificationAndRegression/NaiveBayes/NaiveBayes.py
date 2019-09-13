# coding=utf-8
"""Naive Bayes Text classification."""

from sklearn import datasets, naive_bayes, model_selection, metrics, feature_extraction


if __name__ == "__main__":
    print("Loading data...")
    data = datasets.fetch_20newsgroups()
    oh = feature_extraction.text.CountVectorizer(stop_words="english", ngram_range=(1, 5), max_df=0.8)
    X = oh.fit_transform(data.data)
    y = data.target
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)

    print("#" * 128)
    print("Naive Bayes Multinomial")
    print("Fitting model...")
    nb = naive_bayes.MultinomialNB()
    nb.fit(X_train, y_train)

    print("Evaluating model...")
    print(metrics.classification_report(y_test, nb.predict(X_test)))
    for r in metrics.confusion_matrix(y_test, nb.predict(X_test)):
        print(r.tolist())
