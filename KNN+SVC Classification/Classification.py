from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = datasets.load_breast_cancer()

X = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

results = data.target_names

def find_best_model(x_train, x_test, y_train, y_test):
    best = (0, None)
    for num in range(30):
        model = KNeighborsClassifier(n_neighbors=num+1)

        model.fit(x_train, y_train)

        acc = model.score(x_test, y_test)

        if acc > best[0]:
            best = (acc, model)
    print('Best: ', best[0])
    return best

def support_vector_classifier_best_model(x_train, x_test, y_train, y_test):
    kernels = ['linear']
    current = 0
    best = (0, None)
    C = 20
    for num in range(C*len(kernels)):
        if num > (current+1)*C:
            current += 1
        model = SVC(C=num+1, kernel=kernels[current])
        model.fit(x_train, y_train)

        prediction = model.predict(x_test)

        acc = accuracy_score(y_test, prediction)
        if acc > best[0]:
            best = (acc, model)

    print('Best: ', best[0])
    return best

Kmodel = find_best_model(x_train, x_test, y_train, y_test)[1]

SVCmodel = support_vector_classifier_best_model(x_train, x_test, y_train, y_test)[1]

for Kprediction, result, SVCprediction in zip(Kmodel.predict(x_test), y_test, SVCmodel.predict(x_test)):
    if Kprediction != SVCprediction:
        print(f"""
        KNN Prediction = {results[Kprediction]}
        SVC Prediction = {results[SVCprediction]}
        Result = {results[result]}
        """)
