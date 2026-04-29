"""
Custom CNN Model for Brain Tumor Classification
Based on: Ishfaq et al. (2025) - Scientific Reports

IMPORTANT: This version uses ALREADY PREPROCESSED AND AUGMENTED data
No additional augmentation is applied during training

Architecture Details:
- 5 Convolutional layers (32, 64, 128, 256, 512 filters)
- Max pooling after each conv layer
- 3 Fully connected layers (1024, 512, 10)
- ReLU activation
- Input size: 224x224x3
- Output: 10 classes
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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


class CustomCNN:
    """
    Custom CNN Model for Brain Tumor Classification
    """

    def __init__(self, input_shape=(224, 224, 3), num_classes=10):
        """
        Initialize the Custom CNN model

        Args:
            input_shape: Input image dimensions (height, width, channels)
            num_classes: Number of tumor classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.class_names = None

    def build_model(self):
        """
        Build the Custom CNN architecture as described in the paper
        """
        model = models.Sequential(name='Custom_CNN')

        # Convolutional Layer 1: 32 filters, 3x3 kernel
        model.add(layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same',
                                activation='relu', input_shape=self.input_shape, name='conv1'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1'))

        # Convolutional Layer 2: 64 filters, 3x3 kernel
        model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same',
                                activation='relu', name='conv2'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2'))

        # Convolutional Layer 3: 128 filters, 3x3 kernel
        model.add(layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same',
                                activation='relu', name='conv3'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3'))

        # Convolutional Layer 4: 256 filters, 3x3 kernel
        model.add(layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same',
                                activation='relu', name='conv4'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool4'))

        # Convolutional Layer 5: 512 filters, 3x3 kernel
        model.add(layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same',
                                activation='relu', name='conv5'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool5'))

        # Flatten
        model.add(layers.Flatten(name='flatten'))

        # Fully Connected Layer 1: 1024 units
        model.add(layers.Dense(1024, activation='relu', name='fc1'))
        model.add(layers.Dropout(0.5, name='dropout1'))

        # Fully Connected Layer 2: 512 units
        model.add(layers.Dense(512, activation='relu', name='fc2'))
        model.add(layers.Dropout(0.5, name='dropout2'))

        # Output Layer: num_classes units with softmax
        model.add(layers.Dense(self.num_classes, activation='softmax', name='fc3'))

        self.model = model
        return model

    def compile_model(self, learning_rate=0.001):
        """
        Compile the model with optimizer and loss function

        Args:
            learning_rate: Learning rate for Adam optimizer
        """
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )

        print("✅ Model compiled successfully")
        print(f"   Optimizer: Adam (lr={learning_rate})")
        print(f"   Loss: Categorical Crossentropy")
        print(f"   Metrics: Accuracy, Precision, Recall")

    def get_model_summary(self):
        """Print model architecture summary"""
        return self.model.summary()

    def count_parameters(self):
        """Count total and trainable parameters"""
        total_params = self.model.count_params()
        trainable_params = sum([np.prod(v.shape) for v in self.model.trainable_weights])

        print(f"\n📊 Model Parameters:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Non-trainable parameters: {total_params - trainable_params:,}")

        return total_params, trainable_params


class BrainTumorTrainer:
    """
    Training pipeline for brain tumor classification
    """

    def __init__(self, data_dir, model, batch_size=64, image_size=(224, 224)):
        """
        Initialize trainer

        Args:
            data_dir: Path to data directory (should contain train/val/test folders)
            model: CustomCNN model instance
            batch_size: Batch size for training
            image_size: Input image size
        """
        self.data_dir = Path(data_dir)
        self.model = model
        self.batch_size = batch_size
        self.image_size = image_size

        self.train_dir = self.data_dir / 'train'
        self.val_dir = self.data_dir / 'val'
        self.test_dir = self.data_dir / 'test'

        # Data generators
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None

    def setup_data_generators(self):
        """
        Setup data generators WITHOUT augmentation
        (Data is already preprocessed and augmented)
        """
        print("\n" + "=" * 80)
        print("📊 SETTING UP DATA GENERATORS")
        print("=" * 80)
        print("⚠️  Using PRE-AUGMENTED data - No additional augmentation applied")
        print("=" * 80)

        # All generators only rescale (data already augmented)
        train_datagen = ImageDataGenerator(rescale=1./255)
        val_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)

        # Create generators
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

        # Store class names
        self.model.class_names = list(self.train_generator.class_indices.keys())

        print(f"\n📈 Dataset Statistics:")
        print(f"   Training samples: {self.train_generator.samples}")
        print(f"   Validation samples: {self.val_generator.samples}")
        print(f"   Test samples: {self.test_generator.samples}")
        print(f"   Number of classes: {self.train_generator.num_classes}")
        print(f"   Class names: {self.model.class_names}")
        print("=" * 80)

    def train(self, epochs=30, save_dir='models'):
        """
        Train the model

        Args:
            epochs: Number of training epochs
            save_dir: Directory to save model and results
        """
        # Create save directory
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

        print("\n" + "=" * 80)
        print("🚀 STARTING TRAINING - CUSTOM CNN")
        print("=" * 80)
        print(f"Epochs: {epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Save directory: {save_dir}")
        print("=" * 80)

        # Callbacks
        callbacks = [
            # Save best model
            ModelCheckpoint(
                filepath=str(save_dir / 'custom_cnn_best.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),

            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),

            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),

            # CSV logger
            CSVLogger(
                filename=str(save_dir / 'training_history.csv'),
                separator=',',
                append=False
            )
        ]

        # Train model
        self.model.history = self.model.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            callbacks=callbacks,
            verbose=1
        )

        print("\n✅ Training completed!")

        # Save final model
        final_model_path = save_dir / 'custom_cnn_final.h5'
        self.model.model.save(final_model_path)
        print(f"💾 Final model saved to: {final_model_path}")

        return self.model.history

    def plot_training_history(self, save_dir='models'):
        """
        Plot training history
        """
        history = self.model.history.history
        save_dir = Path(save_dir)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Accuracy
        axes[0, 0].plot(history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0, 0].plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[0, 0].set_title('Custom CNN: Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # Loss
        axes[0, 1].plot(history['loss'], label='Train Loss', linewidth=2)
        axes[0, 1].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 1].set_title('Custom CNN: Model Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # Precision
        axes[1, 0].plot(history['precision'], label='Train Precision', linewidth=2)
        axes[1, 0].plot(history['val_precision'], label='Val Precision', linewidth=2)
        axes[1, 0].set_title('Custom CNN: Model Precision', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        # Recall
        axes[1, 1].plot(history['recall'], label='Train Recall', linewidth=2)
        axes[1, 1].plot(history['val_recall'], label='Val Recall', linewidth=2)
        axes[1, 1].set_title('Custom CNN: Model Recall', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()

        plot_path = save_dir / 'training_history.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training history plot saved to: {plot_path}")
        plt.show()

    def evaluate_on_test(self, save_dir='models'):
        """
        Evaluate model on test set
        """
        print("\n" + "=" * 80)
        print("🧪 EVALUATING ON TEST SET - CUSTOM CNN")
        print("=" * 80)

        save_dir = Path(save_dir)

        # Evaluate
        test_loss, test_accuracy, test_precision, test_recall = self.model.model.evaluate(
            self.test_generator,
            verbose=1
        )

        # Calculate F1 score
        test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall + 1e-7)

        print(f"\n📊 Test Results:")
        print(f"   Accuracy:  {test_accuracy*100:.2f}%")
        print(f"   Precision: {test_precision*100:.2f}%")
        print(f"   Recall:    {test_recall*100:.2f}%")
        print(f"   F1-Score:  {test_f1*100:.2f}%")
        print(f"   Loss:      {test_loss:.4f}")

        # Get predictions
        print("\n🔮 Generating predictions...")
        self.test_generator.reset()
        predictions = self.model.model.predict(self.test_generator, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = self.test_generator.classes

        # Classification report
        print("\n📋 Classification Report:")
        print("=" * 80)
        report = classification_report(
            y_true, y_pred,
            target_names=self.model.class_names,
            digits=4
        )
        print(report)

        # Save classification report
        report_path = save_dir / 'classification_report.txt'
        with open(report_path, 'w') as f:
            f.write("Custom CNN Classification Report\n")
            f.write("=" * 80 + "\n")
            f.write(report)
        print(f"💾 Classification report saved to: {report_path}")

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.model.class_names,
                    yticklabels=self.model.class_names,
                    cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - Custom CNN', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        cm_path = save_dir / 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"📊 Confusion matrix saved to: {cm_path}")
        plt.show()

        # Save test results
        results = {
            'model': 'Custom CNN',
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
    """
    Main execution function
    """
    print("\n" + "=" * 80)
    print("🧠 CUSTOM CNN - BRAIN TUMOR CLASSIFICATION")
    print("Based on: Ishfaq et al. (2025) - Scientific Reports")
    print("Using PRE-AUGMENTED Data")
    print("=" * 80)

    # Configuration
    DATA_DIR = '/content/drive/MyDrive/Project Brain tumer classification/data'
    SAVE_DIR = '/content/drive/MyDrive/Project Brain tumer classification/custom_cnn_results'

    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32 ## 64
    EPOCHS = 30
    LEARNING_RATE = 0.001
    NUM_CLASSES = 10

    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"\n❌ Error: Data directory '{DATA_DIR}' not found!")
        print("Please check the path")
        return

    # Verify train/val/test subdirectories
    required_dirs = ['train', 'val', 'test']
    for subdir in required_dirs:
        full_path = os.path.join(DATA_DIR, subdir)
        if not os.path.exists(full_path):
            print(f"❌ Error: '{subdir}' directory not found in {DATA_DIR}")
            return
        print(f"✅ Found: {full_path}")

    # Create model
    print("\n🏗️  Building Custom CNN model...")
    cnn_model = CustomCNN(
        input_shape=(*IMAGE_SIZE, 3),
        num_classes=NUM_CLASSES
    )

    cnn_model.build_model()
    cnn_model.compile_model(learning_rate=LEARNING_RATE)

    # Print model summary
    print("\n📋 Model Architecture:")
    print("=" * 80)
    cnn_model.get_model_summary()
    cnn_model.count_parameters()

    # Create trainer
    trainer = BrainTumorTrainer(
        data_dir=DATA_DIR,
        model=cnn_model,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE
    )

    # Setup data generators (NO augmentation - data already augmented)
    trainer.setup_data_generators()

    # Train model
    history = trainer.train(epochs=EPOCHS, save_dir=SAVE_DIR)

    # Plot training history
    trainer.plot_training_history(save_dir=SAVE_DIR)

    # Evaluate on test set
    trainer.evaluate_on_test(save_dir=SAVE_DIR)

    print("\n" + "=" * 80)
    print("✅ TRAINING AND EVALUATION COMPLETE!")
    print("=" * 80)
    print(f"📂 All results saved to: {SAVE_DIR}/")
    print("   - custom_cnn_best.h5 (best model)")
    print("   - custom_cnn_final.h5 (final model)")
    print("   - training_history.csv")
    print("   - training_history.png")
    print("   - confusion_matrix.png")
    print("   - classification_report.txt")
    print("   - test_results.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
