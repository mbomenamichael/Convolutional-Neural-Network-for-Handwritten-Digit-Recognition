import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape data to fit CNN input (28x28x1)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Enhanced Data Augmentation to prevent overfitting
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2
)
datagen.fit(x_train)

# Build Enhanced CNN model
model = Sequential([
    Conv2D(64, (5,5), activation='relu', padding='same', input_shape=(28,28,1)),
    BatchNormalization(),
    MaxPooling2D((3,3)),
    
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    
    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.6),
    Dense(10, activation='softmax')
])

# Compile the model with L2 Regularization
from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model using enhanced data augmentation
batch_size = 32
epochs = 15
history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                    validation_data=(x_test, y_test),
                    epochs=epochs)

# Save the trained model
model.save("model10_15e_tf212.h5")
print("Model Successfully Saved")

# Evaluate on test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'Final Test Accuracy: {test_acc * 100:.2f}%')

# Visualize Training Performance
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title("Loss Curve")

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title("Accuracy Curve")
plt.show()

# Check Misclassified Samples
outputs = model.predict(x_test)
labels_predicted = np.argmax(outputs, axis=1)
true_labels = np.argmax(y_test, axis=1)

# Find misclassified samples
misclassified_indices = np.where(labels_predicted != true_labels)[0]

# Plot first few misclassified samples
fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for i, idx in enumerate(misclassified_indices[:10]):
    ax = axes[i // 5, i % 5]
    ax.imshow(x_test[idx].reshape(28,28), cmap='gray_r')
    ax.set_title(f"True: {true_labels[idx]} \nPred: {labels_predicted[idx]}")
    ax.axis("off")

plt.show()
