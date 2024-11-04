import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Paths to your dataset
train_data_dir = r'C:\Users\weake\OneDrive\Desktop\plant\plant affect detection\dataset\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train'
valid_data_dir = r'C:\Users\weake\OneDrive\Desktop\plant\plant affect detection\dataset\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\valid'

# Parameters for data loading
img_height = 128
img_width = 128
batch_size = 32
num_classes = 38  # Number of classes (adjust accordingly)

# Data augmentation for training dataset
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Data augmentation for validation dataset
valid_datagen = ImageDataGenerator()

# Load training data with augmentation
train_dataset = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Load validation data
valid_dataset = valid_datagen.flow_from_directory(
    valid_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Define the model (using MobileNetV2)
base_model = tf.keras.applications.MobileNetV2(input_shape=(img_height, img_width, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False  # Freeze the base model layers

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=10
)

# Evaluate the model
loss, accuracy = model.evaluate(valid_dataset)
print(f'Validation Loss: {loss}, Validation Accuracy: {accuracy}')

# Save the trained model
saved_model_dir = 'plant_disease_model_saved'
model.save(saved_model_dir)

# Convert to TFLite model with quantization
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Representative dataset function for quantization
def representative_dataset():
    for data in valid_dataset:
        yield [data[0]]

converter.representative_dataset = representative_dataset
tflite_model = converter.convert()

# Save the TFLite model
tflite_model_path = 'plant_disease_model_optimized.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"Model successfully converted to {tflite_model_path}")
