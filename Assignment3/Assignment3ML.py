from sklearn import datasets
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt


# Data Loading from A2) 

def load_classification_data():
    cmc = datasets.fetch_openml(data_id=23, as_frame=True)
    nominal_cols = [i for i, dtype in enumerate(cmc.data.dtypes) if str(dtype) == 'category']
    ct = ColumnTransformer([("encoder", OneHotEncoder(sparse_output=False), nominal_cols)], remainder="passthrough")
    encoded_data = ct.fit_transform(cmc.data)


    scaler = ColumnTransformer([("scaler", MinMaxScaler(), list(range(encoded_data.shape[1])))])
    X = scaler.fit_transform(encoded_data)

    enc = OneHotEncoder(sparse_output=False)
    
    y = enc.fit_transform([[x] for x in cmc.target])

    return X, y


def load_regression_data():
    munich = datasets.fetch_openml(data_id=46772, as_frame=True)
    data = munich.data.drop(columns=["rentsqm"], errors="ignore")

    nominal_cols = [i for i, dtype in enumerate(data.dtypes) if str(dtype) == 'category']


    ct = ColumnTransformer([("encoder", OneHotEncoder(sparse_output=False), nominal_cols)], remainder="passthrough")
    encoded_data = ct.fit_transform(data)


    scaler = ColumnTransformer([("scaler", MinMaxScaler(), list(range(encoded_data.shape[1])))])
    X = scaler.fit_transform(encoded_data)
    y = munich.target.astype(float)

    return X, y


# Task 1: Regression 

def run_regression():
    print("\n===== TASK 1: REGRESSION (Munich Rent Index) =====\n")

    X, y = load_regression_data()
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

    n_features = x_tr.shape[1]
    cb = EarlyStopping(monitor="val_mse", patience=20, restore_best_weights=True)

    # Model 1: 2 hidden layers, sigmoid
    nn1 = Sequential()
    nn1.add(Input((n_features,)))
    nn1.add(Dense(64, activation="sigmoid"))
    nn1.add(Dense(32, activation="sigmoid"))

    nn1.add(Dense(1))
    nn1.compile(optimizer="adam", loss="mse", metrics=["mse"])
    h1 = nn1.fit(x_tr, y_tr, epochs=300, validation_data=(x_val, y_val), callbacks=[cb], verbose=0)

    plt.figure()
    plt.plot(h1.history["mse"], label="Training")
    plt.plot(h1.history["val_mse"], label="Validation")


    plt.xlabel("Epochs")
    plt.ylabel("MSE")

    plt.title("Regression - 2 Hidden Layers, Sigmoid")
    plt.legend()
    plt.show()

    # Model 2: ll relu
    nn2 = Sequential()
    nn2.add(Input((n_features,)))

    nn2.add(Dense(64, activation="relu"))
    nn2.add(Dense(32, activation="relu"))

    nn2.add(Dense(1))
    nn2.compile(optimizer="adam", loss="mse", metrics=["mse"])
    h2 = nn2.fit(x_tr, y_tr, epochs=300, validation_data=(x_val, y_val), callbacks=[cb], verbose=0)

    plt.figure()
    plt.plot(h2.history["mse"], label="Training")

    plt.plot(h2.history["val_mse"], label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")

    plt.title("Regression - 2 Hidden Layers, ReLU")
    plt.legend()
    plt.show()

    # Model 3: 15 hidden layers.....sigmoid
    nn3 = Sequential()
    nn3.add(Input((n_features,)))
    for _ in range(15):
        nn3.add(Dense(32, activation="sigmoid"))
    nn3.add(Dense(1))


    nn3.compile(optimizer="adam", loss="mse", metrics=["mse"])
    
    h3 = nn3.fit(x_tr, y_tr, epochs=300, validation_data=(x_val, y_val), callbacks=[cb], verbose=0)

    plt.figure()
    plt.plot(h3.history["mse"], label="Training")
    plt.plot(h3.history["val_mse"], label="Validation")
    plt.xlabel("Epochs")

    plt.ylabel("MSE")
    plt.title("Regression - 15 Hidden Layers, Sigmoid")
    plt.legend()
    plt.show()

    # Model 4o: 15 hidden layers,.., relu
    nn4 = Sequential()
    nn4.add(Input((n_features,)))
    for _ in range(15):
        nn4.add(Dense(32, activation="relu"))
    nn4.add(Dense(1))



    nn4.compile(optimizer="adam", loss="mse", metrics=["mse"])
    h4 = nn4.fit(x_tr, y_tr, epochs=300, validation_data=(x_val, y_val), callbacks=[cb], verbose=0)

    plt.figure()
    plt.plot(h4.history["mse"], label="Training")
    plt.plot(h4.history["val_mse"], label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.title("Regression - 15 Hidden Layers, ReLU")
    plt.legend()
    plt.show()





    # Test results
    r1 = nn1.evaluate(x_test, y_test, verbose=0)
    r2 = nn2.evaluate(x_test, y_test, verbose=0)
    r3 = nn3.evaluate(x_test, y_test, verbose=0)
    r4 = nn4.evaluate(x_test, y_test, verbose=0)

    print("\n--- Regression Test MSE Table ---")

    print(f"2 layers  + Sigmoid : {round(r1[1], 2)}")
    print(f"2 layers  + ReLU    : {round(r2[1], 2)}")
    print(f"15 layers + Sigmoid : {round(r3[1], 2)}")
    print(f"15 layers + ReLU    : {round(r4[1], 2)}")


#  Task 2: Classification ---

def run_classification():
    print("\n===== TASK 2: CLASSIFICATION (CMC Dataset) =====\n")

    X, y = load_classification_data()
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    x_tr, x_val, y_tr, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

    n_features = x_tr.shape[1]
    n_classes = y.shape[1]
    cb = EarlyStopping(monitor="val_accuracy", patience=20, restore_best_weights=True)

    # Model 1: 2 hidden layers, sigmoid
    nn1 = Sequential()
    nn1.add(Input((n_features,)))
    nn1.add(Dense(64, activation="sigmoid"))

    nn1.add(Dense(32, activation="sigmoid"))
    nn1.add(Dense(n_classes, activation="softmax"))

    nn1.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    h1 = nn1.fit(x_tr, y_tr, epochs=300, validation_data=(x_val, y_val), callbacks=[cb], verbose=0)

    plt.figure()
    plt.plot(h1.history["accuracy"], label="Training")
    plt.plot(h1.history["val_accuracy"], label="Validation")
    plt.xlabel("Epochs")

    plt.ylabel("Accuracy")
    plt.title("Classification - 2 Hidden Layers, Sigmoid")
    plt.legend()

    plt.show()

    # Model 2: 2 hidden layers, relu
    nn2 = Sequential()
    nn2.add(Input((n_features,)))
    nn2.add(Dense(64, activation="relu"))
    nn2.add(Dense(32, activation="relu"))

    nn2.add(Dense(n_classes, activation="softmax"))
    nn2.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    h2 = nn2.fit(x_tr, y_tr, epochs=300, validation_data=(x_val, y_val), callbacks=[cb], verbose=0)

    plt.figure()
    plt.plot(h2.history["accuracy"], label="Training")

    plt.plot(h2.history["val_accuracy"], label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    plt.title("Classification - 2 Hidden Layers, ReLU")
    plt.legend()

    plt.show()

    # Model 3: 15 hidden layers, sigmoid
    nn3 = Sequential()
    nn3.add(Input((n_features,)))
    for _ in range(15):
        nn3.add(Dense(32, activation="sigmoid"))

    nn3.add(Dense(n_classes, activation="softmax"))
    nn3.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    h3 = nn3.fit(x_tr, y_tr, epochs=300, validation_data=(x_val, y_val), callbacks=[cb], verbose=0)

    plt.figure()
    plt.plot(h3.history["accuracy"], label="Training")
    plt.plot(h3.history["val_accuracy"], label="Validation")

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    plt.title("Classification - 15 Hidden Layers, Sigmoid")
    plt.legend()
    plt.show()

    # Model 4: 15 hidden layers, reluuu
    nn4 = Sequential()
    nn4.add(Input((n_features,)))
    for _ in range(15):
        nn4.add(Dense(32, activation="relu"))
    nn4.add(Dense(n_classes, activation="softmax"))


    nn4.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    h4 = nn4.fit(x_tr, y_tr, epochs=300, validation_data=(x_val, y_val), callbacks=[cb], verbose=0)

    plt.figure()
    plt.plot(h4.history["accuracy"], label="Training")
    plt.plot(h4.history["val_accuracy"], label="Validation")
    plt.xlabel("Epochs")

    plt.ylabel("Accuracy")
    plt.title("Classification - 15 Hidden Layers, ReLU")
    plt.legend()

    plt.show()

    # Test results
    r1 = nn1.evaluate(x_test, y_test, verbose=0)
    r2 = nn2.evaluate(x_test, y_test, verbose=0)
    r3 = nn3.evaluate(x_test, y_test, verbose=0)
    r4 = nn4.evaluate(x_test, y_test, verbose=0)





    print("\n--- Classification Test Accuracy Table ---")
    print(f"2 layers  + Sigmoid : {round(r1[1], 2)}")
    print(f"2 layers  + ReLU    : {round(r2[1], 2)}")
    print(f"15 layers + Sigmoid : {round(r3[1], 2)}")
    print(f"15 layers + ReLU    : {round(r4[1], 2)}")


# main func for runn

def main():
    run_regression()
    run_classification()


main()