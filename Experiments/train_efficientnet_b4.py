"""
EfficientNet-B4 Transfer Learning for Brain Tumor Classification
Based on: Ishfaq et al. (2025) - Scientific Reports

IMPORTANT: This version uses ALREADY PREPROCESSED AND AUGMENTED data
No additional augmentation is applied during training

Model Specifications:
- Base: EfficientNet-B4 (pre-trained on ImageNet)
- Input size: 224x224x3
- Fine-tuning: All layers trainable
- Output: 10 classes
- Expected Test Accuracy: 99.7%
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


class EfficientNetB4Classifier:
    """
    EfficientNet-B4 Transfer Learning Model
    """

    def __init__(self, input_shape=(224, 224, 3), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.class_names = None

    def build_model(self, trainable=True):
        # Load pre-trained EfficientNet-B4
        base_model = EfficientNetB4(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape,
            pooling='avg'
        )

        base_model.trainable = trainable

        # Build complete model
        inputs = keras.Input(shape=self.input_shape)
        x = base_model(x=inputs, training=trainable)
        
        # Classification head
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu', name='fc1')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax', name='predictions')(x)

        self.model = keras.Model(inputs, outputs, name='EfficientNetB4_Classifier')
        return self.model

    def compile_model(self, learning_rate=0.001):
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )

        print("✅ EfficientNet-B4 model compiled successfully")
        print(f"   Optimizer: Adam (lr={learning_rate})")
        print(f"   Loss: Categorical Crossentropy")

    def get_model_summary(self):
        return self.model.summary()

    def count_parameters(self):
        total_params = self.model.count_params()
        trainable_params = sum([np.prod(v.shape) for v in self.model.trainable_weights])

        print(f"\n📊 Model Parameters:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        return total_params, trainable_params


class BrainTumorTrainer:
    def __init__(self, data_dir, model, batch_size=32, image_size=(224, 224)):
        self.data_dir = Path(data_dir)
        self.model = model
        self.batch_size = batch_size
        self.image_size = image_size

        self.train_dir = self.data_dir / 'train'
        self.val_dir = self.data_dir / 'val'
        self.test_dir = self.data_dir / 'test'

        self.train_generator = None
        self.val_generator = None
        self.test_generator = None

    def setup_data_generators(self):
        """Setup data generators WITHOUT augmentation (data already augmented)"""
        print("\n" + "=" * 80)
        print("📊 SETTING UP DATA GENERATORS")
        print("=" * 80)
        print("⚠️  Using PRE-AUGMENTED data - No additional augmentation applied")
        print("=" * 80)

        # All generators only rescale
        train_datagen = ImageDataGenerator(rescale=1./255)
        val_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)

        self.train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True,
            seed=42
        )

        self.val_generator = val_datagen.flow_from_directory(
            self.val_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )

        self.test_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )

        self.model.class_names = list(self.train_generator.class_indices.keys())

        print(f"\n📈 Dataset Statistics:")
        print(f"   Training samples: {self.train_generator.samples}")
        print(f"   Validation samples: {self.val_generator.samples}")
        print(f"   Test samples: {self.test_generator.samples}")
        print(f"   Number of classes: {self.train_generator.num_classes}")
        print(f"   Class names: {self.model.class_names}")
        print("=" * 80)

    def train(self, epochs=15, save_dir='models'):
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

        print("\n" + "=" * 80)
        print("🚀 STARTING TRAINING - EfficientNet-B4")
        print("=" * 80)
        print(f"Epochs: {epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Save directory: {save_dir}")
        print("=" * 80)

        callbacks = [
            ModelCheckpoint(
                filepath=str(save_dir / 'efficientnet_b4_best.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            CSVLogger(
                filename=str(save_dir / 'training_history.csv'),
                separator=',',
                append=False
            )
        ]

        self.model.history = self.model.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            callbacks=callbacks,
            verbose=1
        )

        print("\n✅ Training completed!")

        final_model_path = save_dir / 'efficientnet_b4_final.h5'
        self.model.model.save(final_model_path)
        print(f"💾 Final model saved to: {final_model_path}")

        return self.model.history

    def plot_training_history(self, save_dir='models'):
        history = self.model.history.history
        save_dir = Path(save_dir)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Accuracy
        axes[0, 0].plot(history['accuracy'], label='Train', linewidth=2)
        axes[0, 0].plot(history['val_accuracy'], label='Val', linewidth=2)
        axes[0, 0].set_title('EfficientNet-B4: Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # Loss
        axes[0, 1].plot(history['loss'], label='Train', linewidth=2)
        axes[0, 1].plot(history['val_loss'], label='Val', linewidth=2)
        axes[0, 1].set_title('EfficientNet-B4: Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # Precision
        axes[1, 0].plot(history['precision'], label='Train', linewidth=2)
        axes[1, 0].plot(history['val_precision'], label='Val', linewidth=2)
        axes[1, 0].set_title('EfficientNet-B4: Precision', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        # Recall
        axes[1, 1].plot(history['recall'], label='Train', linewidth=2)
        axes[1, 1].plot(history['val_recall'], label='Val', linewidth=2)
        axes[1, 1].set_title('EfficientNet-B4: Recall', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plot_path = save_dir / 'training_history.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training history saved to: {plot_path}")
        plt.show()

    def evaluate_on_test(self, save_dir='models'):
        print("\n" + "=" * 80)
        print("🧪 EVALUATING ON TEST SET - EfficientNet-B4")
        print("=" * 80)

        save_dir = Path(save_dir)

        test_loss, test_accuracy, test_precision, test_recall = self.model.model.evaluate(
            self.test_generator,
            verbose=1
        )

        test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall + 1e-7)

        print(f"\n📊 Test Results:")
        print(f"   Accuracy:  {test_accuracy*100:.2f}%")
        print(f"   Precision: {test_precision*100:.2f}%")
        print(f"   Recall:    {test_recall*100:.2f}%")
        print(f"   F1-Score:  {test_f1*100:.2f}%")
        print(f"   Loss:      {test_loss:.4f}")

        print("\n🔮 Generating predictions...")
        self.test_generator.reset()
        predictions = self.model.model.predict(self.test_generator, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = self.test_generator.classes

        print("\n📋 Classification Report:")
        print("=" * 80)
        report = classification_report(y_true, y_pred, target_names=self.model.class_names, digits=4)
        print(report)

        report_path = save_dir / 'classification_report.txt'
        with open(report_path, 'w') as f:
            f.write("EfficientNet-B4 Classification Report\n" + "=" * 80 + "\n" + report)
        print(f"💾 Classification report saved to: {report_path}")

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                    xticklabels=self.model.class_names,
                    yticklabels=self.model.class_names,
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - EfficientNet-B4', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        cm_path = save_dir / 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"📊 Confusion matrix saved to: {cm_path}")
        plt.show()

        results = {
            'model': 'EfficientNet-B4',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'test_accuracy': float(test_accuracy),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'test_f1_score': float(test_f1),
            'test_loss': float(test_loss),
            'confusion_matrix': cm.tolist(),
            'class_names': self.model.class_names
        }

        results_path = save_dir / 'test_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Test results saved to: {results_path}")
        print("=" * 80)

        return test_accuracy, test_precision, test_recall, test_f1


def main():
    print("\n" + "=" * 80)
    print("🧠 EfficientNet-B4 - BRAIN TUMOR CLASSIFICATION")
    print("Based on: Ishfaq et al. (2025) - Scientific Reports")
    print("Using PRE-AUGMENTED Data")
    print("Expected Test Accuracy: 99.7%")
    print("=" * 80)

    # Configuration
    DATA_DIR = '/content/drive/MyDrive/Project Brain tumer classification/data'
    SAVE_DIR = '/content/drive/MyDrive/Project Brain tumer classification/efficientnet_b4_results'

    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 0.001
    NUM_CLASSES = 10

    if not os.path.exists(DATA_DIR):
        print(f"\n❌ Error: Data directory '{DATA_DIR}' not found!")
        return

    # Verify directories
    for subdir in ['train', 'val', 'test']:
        path = os.path.join(DATA_DIR, subdir)
        if os.path.exists(path):
            print(f"✅ Found: {path}")
        else:
            print(f"❌ Missing: {path}")
            return

    print("\n🏗️  Building EfficientNet-B4 model...")
    model = EfficientNetB4Classifier(input_shape=(*IMAGE_SIZE, 3), num_classes=NUM_CLASSES)
    model.build_model(trainable=True)
    model.compile_model(learning_rate=LEARNING_RATE)

    print("\n📋 Model Architecture:")
    print("=" * 80)
    model.get_model_summary()
    model.count_parameters()

    trainer = BrainTumorTrainer(data_dir=DATA_DIR, model=model, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)
    trainer.setup_data_generators()
    trainer.train(epochs=EPOCHS, save_dir=SAVE_DIR)
    trainer.plot_training_history(save_dir=SAVE_DIR)
    trainer.evaluate_on_test(save_dir=SAVE_DIR)

    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print(f"📂 Results saved to: {SAVE_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
