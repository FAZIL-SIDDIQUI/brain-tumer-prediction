"""
Memory-Efficient Inception-v4 Training
Prevents kernel crashes with:
- Smaller batch size (8 instead of 64)
- Mixed precision
- Memory cleanup
- Frequent checkpointing
"""

import os, gc
import numpy as np
import json
from datetime import datetime
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend as K
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, Callback

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

class MemoryCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        K.clear_session()
        print(f"\n💾 Memory cleaned after epoch {epoch + 1}")

# Configuration
DATA_DIR = '/content/drive/MyDrive/Project Brain tumer classification/data'
SAVE_DIR = '/content/drive/MyDrive/Project Brain tumer classification/inception_v4_results'
IMAGE_SIZE = (299, 299)  # Inception requires 299x299
BATCH_SIZE = 8  # Reduced for memory
EPOCHS = 20
NUM_CLASSES = 10

print("\n" + "=" * 80)
print("🧠 MEMORY-EFFICIENT Inception-v4 TRAINING")
print("=" * 80)
print(f"Batch size: {BATCH_SIZE} (reduced for memory efficiency)")
print(f"Image size: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} (Inception requires 299x299)")
print("Mixed precision: Enabled")
print("=" * 80)

# Setup mixed precision
policy = keras.mixed_precision.Policy('mixed_float16')
keras.mixed_precision.set_global_policy(policy)

# Build model
print("\n🏗️  Building Inception-v4...")
base_model = InceptionResNetV2(include_top=False, weights='imagenet',
                               input_shape=(*IMAGE_SIZE, 3), pooling='avg')
base_model.trainable = True

inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
x = base_model(inputs, training=True)
x = layers.Dropout(0.5)(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = keras.Model(inputs, outputs, name='InceptionV4')
model.compile(optimizer=keras.optimizers.Adam(0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])

print("✅ Model compiled")

# Data generators
print("\n📊 Setting up data generators...")
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'), target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True, seed=42)

val_gen = val_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'val'), target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

test_gen = test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'test'), target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

print(f"✅ Train: {train_gen.samples} | Val: {val_gen.samples} | Test: {test_gen.samples}")

# Train
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

callbacks = [
    ModelCheckpoint(f'{SAVE_DIR}/inception_v4_best.h5', monitor='val_accuracy',
                   save_best_only=True, mode='max', verbose=1),
    MemoryCallback(),
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    CSVLogger(f'{SAVE_DIR}/training_history.csv')
]

print("\n🚀 Starting training...")
history = model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen,
                   callbacks=callbacks, verbose=1)

model.save(f'{SAVE_DIR}/inception_v4_final.h5')
print(f"\n✅ Training complete! Model saved to {SAVE_DIR}/")

# Evaluate
print("\n🧪 Evaluating on test set...")
test_loss, test_acc, test_prec, test_rec = model.evaluate(test_gen, verbose=1)
print(f"\n📊 Test Accuracy: {test_acc*100:.2f}%")
print(f"   Precision: {test_prec*100:.2f}%")
print(f"   Recall: {test_rec*100:.2f}%")

# Save results
results = {
    'model': 'Inception-v4',
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'test_accuracy': float(test_acc),
    'test_precision': float(test_prec),
    'test_recall': float(test_rec)
}

with open(f'{SAVE_DIR}/test_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ All results saved to: {SAVE_DIR}/")
