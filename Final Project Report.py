

import os
import shutil
import random
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Data paths
dataset_path = '/content/drive/MyDrive/Colab/dataset/'
output_path = '/content/drive/MyDrive/Colab/output_images/'
train_path = '/content/drive/MyDrive/Colab/train_data/'
test_path = '/content/drive/MyDrive/Colab/test_data/'

# Ensure output directories exist
classes = ['Arcing', 'Corona', 'Looseness', 'Tracking']
for cls in classes:
    os.makedirs(os.path.join(output_path, cls), exist_ok=True)
    os.makedirs(os.path.join(train_path, cls), exist_ok=True)
    os.makedirs(os.path.join(test_path, cls), exist_ok=True)

# Parameters
segment_duration = 0.5  # 0.5 seconds per segment
overlap_duration = 0.2  # Overlap duration
sample_rate = 22050     # Audio sampling rate

# Step 1: Generate Images from .wav Files
for file in os.listdir(dataset_path):
    if file.endswith(".wav"):
        file_path = os.path.join(dataset_path, file)
        class_name = next((cls for cls in classes if cls.lower() in file.lower()), None)
        if not class_name:
            continue  # Skip files with unknown class
             # Load audio file
        audio, sr = librosa.load(file_path, sr=sample_rate)
            # Calculate segment parameters
        segment_samples = int(segment_duration * sr)
        step_samples = int((segment_duration - overlap_duration) * sr)
           # Create segments and save as images
        for i in range(0, len(audio) - segment_samples, step_samples):
            segment = audio[i:i + segment_samples]
             # Generate Mel Spectrogram
            S = librosa.feature.melspectrogram(y=segment, sr=sr)
            S_dB = librosa.power_to_db(S, ref=np.max)
            # Save as black & white image
            plt.figure(figsize=(1, 1), dpi=64)
            librosa.display.specshow(S_dB, cmap='gray_r')
            plt.axis('off')
            img_path = os.path.join(output_path, class_name, f"{file}_segment_{i}.png")
            plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
            plt.close()

# Step 2: Split Data (80% Training, 20% Testing)
for cls in classes:
    class_path = os.path.join(output_path, cls)
    class_files = os.listdir(class_path)
    random.shuffle(class_files)
    split_index = int(0.8 * len(class_files))

    for file in class_files[:split_index]:
        shutil.copy(os.path.join(class_path, file), os.path.join(train_path, cls))
    for file in class_files[split_index:]:
        shutil.copy(os.path.join(class_path, file), os.path.join(test_path, cls))

# Step 3: Data Generators
img_size = (224, 224)
batch_size = 32

datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(
    train_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = datagen.flow_from_directory(
    test_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Step 4: Model Architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(4, activation='softmax')  # 4 output classes: Arcing, Corona, Looseness, Tracking
])

# Step 5: Compile the Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 6: Train the Model
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=test_generator
)

# Step 7: Plot Training Results
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Step 8: Evaluate the Model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Step 9: Confusion Matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
y_true = test_generator.classes

cm = confusion_matrix(y_true, y_pred)
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
display.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
