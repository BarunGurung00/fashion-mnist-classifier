"""
Fashion-MNIST Classification using a Dense Neural Network
Author: Barun
Description: Loads Fashion-MNIST dataset, preprocesses and normalizes images, 
trains a multi-layer deep neural network using TensorFlow/Keras, evaluates 
performance, and visualizes accuracy plots and predictions.
"""

import os
import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard

# Set up logging directory for TensorBoard
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

def main():
    print("--- 1. Loading Fashion-MNIST Dataset ---")
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    
    print(f"Train images shape: {train_images.shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Test images shape: {test_images.shape}")
    print(f"Test labels shape: {test_labels.shape}")

    labels_map = {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot",
    }

    # Visualize a training sample
    training_sample = 1
    print(f"\nVisualizing training sample index {training_sample} (Label: {labels_map[train_labels[training_sample]]})")
    plt.figure(figsize=(4, 4))
    plt.imshow(train_images[training_sample], cmap='gray')
    plt.title(f"Label: {labels_map[train_labels[training_sample]]}")
    plt.axis('off')
    plt.show()

    print("\n--- 2. Preprocessing Data ---")
    # Flatten images from 28x28 to 784 vectors and normalize pixel values between 0 and 1
    x_trainp = train_images.reshape(60000, 784).astype('float32') / 255.0
    x_testp = test_images.reshape(10000, 784).astype('float32') / 255.0

    print(f"{x_trainp.shape[0]} train samples")
    print(f"{x_testp.shape[0]} test samples")

    # One-hot encode output labels
    num_classes = 10
    y_trainp = keras.utils.to_categorical(train_labels, num_classes)
    y_testp = keras.utils.to_categorical(test_labels, num_classes)
    
    print("One-hot encoding complete.")
    print(f"Label T-shirt (index 1) example vector: {y_trainp[1]}")

    print("\n--- 3. Building Neural Network Model ---")
    model = Sequential([
        Dense(32, activation='relu', input_shape=(784,)),
        Dense(16, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer=RMSprop(),
        metrics=['accuracy']
    )
    
    model.summary()

    print("\n--- 4. Training the Model ---")
    epochs = 10
    batch_size = 128

    # Validation split from training data
    x_valp = x_trainp[:10000]
    partial_x_trainp = x_trainp[10000:]
    y_valp = y_trainp[:10000]
    partial_y_trainp = y_trainp[10000:]

    tensorboard_callback = TensorBoard(logdir, histogram_freq=1)

    history = model.fit(
        partial_x_trainp, partial_y_trainp,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_valp, y_valp),
        callbacks=[tensorboard_callback]
    )

    # Plot Training and Validation Accuracy
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, acc, 'r', label='Training acc')
    plt.plot(epochs_range, val_acc, 'b', label='Validation acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    print("\n--- 5. Evaluating on Test Set ---")
    score = model.evaluate(x_testp, y_testp, verbose=0)
    print(f'Test loss: {score[0]:.4f}')
    print(f'Test accuracy: {score[1]:.4f}')

    # Predictions
    ans = model.predict(x_testp)

    # Helper function to predict and visualize a test sample
    def predict_one_image(p_index):
        print(f"\nVerifying prediction for test sample index {p_index}:")
        plt.figure(figsize=(4, 4))
        plt.imshow(test_images[p_index], cmap='gray')
        plt.title(f"True Label: {labels_map[test_labels[p_index]]}")
        plt.axis('off')
        plt.show()

        print("Class Probabilities:")
        for count, a in enumerate(ans[p_index]):
            print(f"  {labels_map[count]}: {round(a * 100, 2)}%")

        print("-" * 30)
        print(f"True Label: {labels_map[test_labels[p_index]]}")

    # Run prediction visualization on sample 0
    predict_one_image(0)

if __name__ == '__main__':
    main()