import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt

def optimized_lenet5():
    model = models.Sequential([
        layers.Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 1), padding='valid', kernel_initializer='he_uniform'),
        layers.AveragePooling2D(2, strides=2),
        layers.BatchNormalization(),
        
        layers.Conv2D(16, (5, 5), activation='relu', padding='valid', kernel_initializer='he_uniform'),
        layers.AveragePooling2D(2, strides=2),
        layers.BatchNormalization(),
        
        layers.Conv2D(120, (5, 5), activation='relu', kernel_initializer='he_uniform'),
        layers.BatchNormalization(),
        
        layers.Flatten(),
        layers.Dense(84, activation='relu', kernel_initializer='he_uniform'),
        layers.BatchNormalization(),
        
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Pad images to 32x32 and add an extra dimension
train_images = np.pad(train_images, ((0, 0), (2, 2), (2, 2)), 'constant')
test_images = np.pad(test_images, ((0, 0), (2, 2), (2, 2)), 'constant')
train_images, test_images = train_images[..., np.newaxis] / 255.0, test_images[..., np.newaxis] / 255.0

# One-hot encode labels
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)


model = optimized_lenet5()
model.summary()

# Train the model
history = model.fit(train_images, train_labels, epochs=10, validation_split=0.1)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_acc*100:.2f}%")




# Plot training & validation accuracy values
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

