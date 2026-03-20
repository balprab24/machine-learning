from sklearn import datasets
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyClassifier, DummyRegressor


# Multi class classificaion cmc

def load_classification_data():
    cmc = datasets.fetch_openml(data_id=23, as_frame=True)
    nominal_cols = [i for i, dtype in enumerate(cmc.data.dtypes) if str(dtype) == 'category']

    ct = ColumnTransformer([("encoder", OneHotEncoder(sparse_output=False), nominal_cols)], remainder="passthrough")
    encoded_data = ct.fit_transform(cmc.data)

    scaler = ColumnTransformer([("scaler", MinMaxScaler(), list(range(encoded_data.shape[1])))])
    scaled_data = scaler.fit_transform(encoded_data)



    return scaled_data, cmc.target


def print_classification_results(name, results):
    print(name)
    print(f"  Mean Accuracy:      {results['test_accuracy'].mean():.2f}")
    print(f"  Mean Training Time: {results['fit_time'].mean():.2f} sec")
    print(f"  Mean Testing Time:  {results['score_time'].mean():.2f} sec\n")


def run_classification(X, y):
    print("\n      TASK 1: MULTI-CLASS CLASSIFICATION (cmc)     \n")

    dt_tuned = GridSearchCV(DecisionTreeClassifier(random_state=0), {"min_samples_leaf": [1, 5, 10, 20, 50]}, cv=10, scoring="accuracy")
    print_classification_results("Decision Tree:", cross_validate(dt_tuned, X, y, cv=10, scoring=["accuracy"]))

    knn_tuned = GridSearchCV(KNeighborsClassifier(), {"n_neighbors": [1, 3, 5, 7, 9]}, cv=10, scoring="accuracy")
    print_classification_results("KNN:", cross_validate(knn_tuned, X, y, cv=10, scoring=["accuracy"]))

    print_classification_results("Multinomial Naive Bayes:", cross_validate(MultinomialNB(), X, y, cv=10, scoring=["accuracy"]))

    print_classification_results("Dummy Classifier:", cross_validate(DummyClassifier(strategy="most_frequent"), X, y, cv=10, scoring=["accuracy"]))


# REgression munich

def load_regression_data():
    munich = datasets.fetch_openml(data_id=46772, as_frame=True)
    data = munich.data.drop(columns=["rentsqm"], errors="ignore")


    nominal_cols = [i for i, dtype in enumerate(data.dtypes) if str(dtype) == 'category']
    ct = ColumnTransformer([("encoder", OneHotEncoder(sparse_output=False), nominal_cols)], remainder="passthrough")


    encoded_data = ct.fit_transform(data)
    scaler = ColumnTransformer([("scaler", MinMaxScaler(), list(range(encoded_data.shape[1])))])
    scaled_data = scaler.fit_transform(encoded_data)


    return scaled_data, munich.target.astype(float)


def print_regression_results(name, results):
    print(name)
    
    print(f"  Mean RMSE:          {(0 - results['test_neg_root_mean_squared_error']).mean():.2f}")
    print(f"  Mean R2:            {results['test_r2'].mean():.2f}")
    print(f"  Mean Training Time: {results['fit_time'].mean():.2f} sec")
    print(f"  Mean Testing Time:  {results['score_time'].mean():.2f} sec\n")


def run_regression(X, y):
    print("\n     TASK 2: REGRESSION (munich-rent-index-1999)    \n")

    dt_tuned = GridSearchCV(DecisionTreeRegressor(random_state=0), {"min_samples_leaf": [1, 5, 10, 20, 50]}, cv=10, scoring="neg_root_mean_squared_error")
    print_regression_results("Decision Tree Regressor:", cross_validate(dt_tuned, X, y, cv=10, scoring=["neg_root_mean_squared_error", "r2"]))

    knn_tuned = GridSearchCV(KNeighborsRegressor(), {"n_neighbors": [1, 3, 5, 7, 9]}, cv=10, scoring="neg_root_mean_squared_error")
    print_regression_results("KNN Regressor:", cross_validate(knn_tuned, X, y, cv=10, scoring=["neg_root_mean_squared_error", "r2"]))



    print_regression_results("Linear Regression:", cross_validate(LinearRegression(), X, y, cv=10, scoring=["neg_root_mean_squared_error", "r2"]))

    print_regression_results("Dummy Regressor:", cross_validate(DummyRegressor(strategy="mean"), X, y, cv=10, scoring=["neg_root_mean_squared_error", "r2"]))


#MAINNNNN
def main():
    X_cls, y_cls = load_classification_data()
    run_classification(X_cls, y_cls)

    X_reg, y_reg = load_regression_data()
    run_regression(X_reg, y_reg)





main()