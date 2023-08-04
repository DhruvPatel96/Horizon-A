
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D,Dropout
from tensorflow.keras.models import Model
from keras.optimizers import Adam
from tensorflow.keras.applications import Xception,ResNet50
from tensorflow.keras.regularizers import l2
#-------------------------------------------------------------------------------
train_data_dir = 'YOUR MAIN TRAINING DIRECTORY PATH'
test_data_dir = 'YOUR MAIN TEST DIRECTORY PATH'
img_width, img_height = 299, 299
batch_size = 64
num_epochs = 50
num_classes = len(os.listdir(train_data_dir))
#-------------------------------------------------------------------------------
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    zoom_range = 0.2,
    rotation_range = 10,
    shear_range = 0.2,
    height_shift_range = 0.1,
    width_shift_range = 0.1
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)
#-------------------------------------------------------------------------------
base_model = Xception(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.2)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base Xception layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

#-------------------------------------------------------------------------------
import requests
import time

while True:
    try:
        requests.get('https://www.google.com')
        print("Kept alive.")

        model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=num_epochs,
            validation_data=test_generator,
            validation_steps=test_generator.samples // batch_size
        )

    except:
        print("Failed to keep alive.")


model.save('soil_Xcep_299_V2.h5')
#-------------------------------------------------------------------------------
import matplotlib.pyplot as plt
# Extracting training and validation metrics from the history
train_loss = history.history['loss']
train_accuracy = history.history['accuracy']
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']

# Calculate bias and variance
bias = 1 - val_accuracy[-1]  # Bias is the difference between 1 and the final validation accuracy
variance = max(val_accuracy) - min(val_accuracy)  # Variance is the difference between the maximum and minimum validation accuracies

# Print Bias and Variance
print(f'Bias: {bias:.4f}')
print(f'Variance: {variance:.4f}')

# Determine if it is high bias, high variance, or neither
if bias > 0.2 and variance < 0.1:
    print('High Bias (Underfitting)')
elif bias < 0.1 and variance > 0.3:
    print('High Variance (Overfitting)')
elif bias > 0.2 and variance > 0.3:
    print('High Bias and High Variance')
else:
    print('Bias and Variance are within acceptable range')

# Plotting Training Loss and Accuracy, and Test Loss and Accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# Plotting Bias and Variance
plt.figure(figsize=(6, 4))
plt.bar(['Bias', 'Variance'], [bias, variance])
plt.xlabel('Metrics')
plt.ylabel('Value')
plt.title('Bias-Variance Tradeoff')
plt.show()