import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Rescaling, Conv2D, Dense, Flatten, MaxPooling2D, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetV2B3
from tensorflow.keras.preprocessing.image import load_img, img_to_array

DATASET_PATH = "A4_dataset"

TASK4_IMAGES = [
    "8808.png",
    "9599.png",
    "9347.png",
    "2694.png",
    "892.png",
]
TASK4_TRUE_AGES = [65, 14, 38, 100, 44]


def load_data(dataset_path=DATASET_PATH):
    training_set = image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="training",
        label_mode="categorical",
        seed=0,
        image_size=(200, 200)
    )
    test_set = image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="validation",
        label_mode="categorical",
        seed=0,
        image_size=(200, 200)
    )
    print("Class names:", training_set.class_names)
    return training_set, test_set


def build_fc_model():
    m = Sequential()
    m.add(Input((200, 200, 3)))
    m.add(Rescaling(1 / 255))
    m.add(Flatten())
    m.add(Dense(128, activation="relu"))
    m.add(Dropout(0.2))
    m.add(Dense(64, activation="relu"))
    m.add(Dropout(0.2))
    m.add(Dense(5, activation="softmax"))
    m.compile(loss="categorical_crossentropy", metrics=["accuracy"])
    m.summary()
    return m


def train_fc_model(training_set, epochs=15):
    m = build_fc_model()
    history = m.fit(training_set, epochs=epochs, verbose=2)
    return m, history


def build_cnn_model():
    m = Sequential()
    m.add(Input((200, 200, 3)))
    m.add(Rescaling(1 / 255))
    m.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Conv2D(64, (3, 3), activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Conv2D(64, (3, 3), activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Flatten())
    m.add(Dense(128, activation="relu"))
    m.add(Dropout(0.2))
    m.add(Dense(5, activation="softmax"))
    m.compile(loss="categorical_crossentropy", metrics=["accuracy"])
    m.summary()
    return m


def train_cnn_model(training_set, epochs=15, save_path="task2_cnn.keras"):
    m = build_cnn_model()
    history = m.fit(training_set, epochs=epochs, verbose=2)
    m.save(save_path)
    print("CNN model saved to:", save_path)
    return m, history


def build_pretrained_model():
    base_model = EfficientNetV2B3(include_top=False)
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    output_layer = Dense(5, activation="softmax")(x)
    m = Model(inputs=base_model.input, outputs=output_layer)
    m.compile(loss="categorical_crossentropy", metrics=["accuracy"])
    m.summary()
    return m


def train_pretrained_model(training_set, epochs=10, save_path="task3_pretrained.keras"):
    m = build_pretrained_model()
    history = m.fit(training_set, epochs=epochs, verbose=2)
    m.save(save_path)
    print("Pretrained model saved to:", save_path)
    return m, history


def evaluate_model(m, test_set, model_name="Model"):
    score = m.evaluate(test_set, verbose=2)
    print("Test accuracy:", round(score[1], 3))
    return score[1]


def predict_image(model, image_file):
    img = load_img(image_file, target_size=(200, 200))
    img_arr = img_to_array(img)
    img_cl = img_arr.reshape(1, 200, 200, 3)
    score = model.predict(img_cl, verbose=0)
    return score[0]


def run_task4(model_fc, model_cnn, model_pretrained, class_names):
    print("\nTASK 4: Age Prediction on New Images")
    print(f"{'Image':<20} {'True Age':<12} {'Model':<12} " + "  ".join([f"{c:>10}" for c in class_names]))
    print("-" * 85)
    for img_path, true_age in zip(TASK4_IMAGES, TASK4_TRUE_AGES):
        for model, name in [(model_fc, "FC"), (model_cnn, "CNN"), (model_pretrained, "Pretrained")]:
            probs = predict_image(model, img_path)
            prob_str = "  ".join([f"{p:>10.3f}" for p in probs])
            label = img_path if name == "FC" else ""
            age_str = str(true_age) if name == "FC" else ""
            print(f"{label:<20} {age_str:<12} {name:<12} {prob_str}")
        print()


def plot_training_accuracy(histories, labels):
    plt.figure(figsize=(10, 6))
    for history, label in zip(histories, labels):
        plt.plot(history.history["accuracy"], label=label, marker='o')
    plt.title("Training Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_accuracy.png")
    print("Training accuracy plot saved to: training_accuracy.png")


def main():
    training_set, test_set = load_data()
    class_names = training_set.class_names

    print("\nTASK 1: Fully-Connected Model")
    model_fc, history_fc = train_fc_model(training_set, epochs=15)
    acc_fc = evaluate_model(model_fc, test_set, "Fully-Connected")
    model_fc.save("task1_fc.keras")

    print("\nTASK 2: CNN Model")
    model_cnn, history_cnn = train_cnn_model(training_set, epochs=15)
    acc_cnn = evaluate_model(model_cnn, test_set, "CNN")

    print("\nTASK 3: Fine-Tuned Pre-Trained Model (EfficientNetV2B3)")
    model_pre, history_pre = train_pretrained_model(training_set, epochs=10)
    acc_pre = evaluate_model(model_pre, test_set, "Fine-Tuned Pretrained")

    plot_training_accuracy(
        [history_fc, history_cnn, history_pre],
        ["Fully-Connected", "CNN", "Fine-Tuned Pretrained"]
    )

    print("\nTest Accuracy Summary")
    print(f"{'Fully-Connected':<25} {acc_fc:.3f}")
    print(f"{'CNN':<25} {acc_cnn:.3f}")
    print(f"{'Fine-Tuned Pretrained':<25} {acc_pre:.3f}")

    run_task4(model_fc, model_cnn, model_pre, class_names)


if __name__ == "__main__":
    main()