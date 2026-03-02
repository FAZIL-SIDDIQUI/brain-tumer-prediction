"""
Memory-Efficient Custom CNN Training with Batch Processing
Prevents Colab kernel crashes by:
1. Processing data in smaller batches
2. Frequent checkpointing
3. Memory cleanup between epochs
4. Reduced batch size
"""

import os
import gc
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, Callback
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


# Memory optimization
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'


class MemoryCallback(Callback):
    """Callback to clear memory after each epoch"""
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        K.clear_session()
        print(f"\n💾 Memory cleaned after epoch {epoch + 1}")


class CustomCNN:
    def __init__(self, input_shape=(224, 224, 3), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        self.class_names = None

    def build_model(self):
        """Build Custom CNN with memory optimization"""
        model = models.Sequential(name='Custom_CNN')

        # Conv layers
        model.add(layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same',
                                activation='relu', input_shape=self.input_shape, name='conv1'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1'))

        model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same',
                                activation='relu', name='conv2'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2'))

        model.add(layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same',
                                activation='relu', name='conv3'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3'))

        model.add(layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same',
                                activation='relu', name='conv4'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool4'))

        model.add(layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same',
                                activation='relu', name='conv5'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool5'))

        # Flatten and Dense layers
        model.add(layers.Flatten(name='flatten'))
        model.add(layers.Dense(1024, activation='relu', name='fc1'))
        model.add(layers.Dropout(0.5, name='dropout1'))
        model.add(layers.Dense(512, activation='relu', name='fc2'))
        model.add(layers.Dropout(0.5, name='dropout2'))
        model.add(layers.Dense(self.num_classes, activation='softmax', name='fc3'))

        self.model = model
        return model

    def compile_model(self, learning_rate=0.001):
        # Mixed precision for memory efficiency
        policy = keras.mixed_precision.Policy('mixed_float16')
        keras.mixed_precision.set_global_policy(policy)
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )

        print("✅ Model compiled with mixed precision (memory efficient)")
        print(f"   Optimizer: Adam (lr={learning_rate})")


class MemoryEfficientTrainer:
    def __init__(self, data_dir, model, batch_size=16, image_size=(224, 224)):
        """
        Initialize trainer with smaller batch size
        
        Args:
            batch_size: Reduced to 16 (from 64) to prevent memory crashes
        """
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
        """Setup memory-efficient data generators"""
        print("\n" + "=" * 80)
        print("📊 SETTING UP MEMORY-EFFICIENT DATA GENERATORS")
        print("=" * 80)
        print("⚠️  Using PRE-AUGMENTED data")
        print(f"⚠️  Batch size: {self.batch_size} (reduced for memory efficiency)")
        print("=" * 80)

        # Only rescaling - data already augmented
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
        print(f"   Batches per epoch: {len(self.train_generator)}")
        print("=" * 80)

    def train(self, epochs=30, save_dir='models'):
        """Train with frequent checkpointing"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

        print("\n" + "=" * 80)
        print("🚀 STARTING MEMORY-EFFICIENT TRAINING")
        print("=" * 80)
        print(f"Epochs: {epochs}")
        print(f"Batch size: {self.batch_size} (memory optimized)")
        print(f"Mixed precision: Enabled")
        print(f"Save directory: {save_dir}")
        print("=" * 80)

        # Callbacks with frequent saving
        callbacks = [
            # Save every epoch (not just best)
            ModelCheckpoint(
                filepath=str(save_dir / 'custom_cnn_checkpoint.h5'),
                monitor='val_accuracy',
                save_best_only=False,
                save_freq='epoch',
                verbose=1
            ),
            
            # Save best model
            ModelCheckpoint(
                filepath=str(save_dir / 'custom_cnn_best.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),

            # Memory cleanup
            MemoryCallback(),

            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),

            # Reduce learning rate
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
                append=True  # Append in case of restart
            )
        ]

        # Check if checkpoint exists (resume training)
        checkpoint_path = save_dir / 'custom_cnn_checkpoint.h5'
        initial_epoch = 0
        
        if checkpoint_path.exists():
            print(f"\n⚠️  Found checkpoint at {checkpoint_path}")
            resume = input("Resume from checkpoint? (y/n): ").strip().lower()
            if resume == 'y':
                print("📥 Loading checkpoint...")
                self.model.model = keras.models.load_model(checkpoint_path)
                # Try to determine which epoch to resume from
                csv_path = save_dir / 'training_history.csv'
                if csv_path.exists():
                    import pandas as pd
                    history_df = pd.read_csv(csv_path)
                    initial_epoch = len(history_df)
                    print(f"✅ Resuming from epoch {initial_epoch}")

        try:
            # Train model
            self.model.history = self.model.model.fit(
                self.train_generator,
                epochs=epochs,
                initial_epoch=initial_epoch,
                validation_data=self.val_generator,
                callbacks=callbacks,
                verbose=1
            )

            print("\n✅ Training completed!")

        except Exception as e:
            print(f"\n⚠️  Training interrupted: {str(e)}")
            print("💾 Progress saved in checkpoint file")
            print("   Run script again to resume training")
            return None

        # Save final model
        final_model_path = save_dir / 'custom_cnn_final.h5'
        self.model.model.save(final_model_path)
        print(f"💾 Final model saved to: {final_model_path}")

        # Clean up memory
        gc.collect()
        K.clear_session()

        return self.model.history

    def plot_training_history(self, save_dir='models'):
        """Plot training history"""
        if self.model.history is None:
            print("⚠️  No training history to plot")
            return

        history = self.model.history.history
        save_dir = Path(save_dir)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Accuracy
        axes[0, 0].plot(history['accuracy'], label='Train', linewidth=2)
        axes[0, 0].plot(history['val_accuracy'], label='Val', linewidth=2)
        axes[0, 0].set_title('Custom CNN: Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # Loss
        axes[0, 1].plot(history['loss'], label='Train', linewidth=2)
        axes[0, 1].plot(history['val_loss'], label='Val', linewidth=2)
        axes[0, 1].set_title('Custom CNN: Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # Precision
        axes[1, 0].plot(history['precision'], label='Train', linewidth=2)
        axes[1, 0].plot(history['val_precision'], label='Val', linewidth=2)
        axes[1, 0].set_title('Custom CNN: Precision', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        # Recall
        axes[1, 1].plot(history['recall'], label='Train', linewidth=2)
        axes[1, 1].plot(history['val_recall'], label='Val', linewidth=2)
        axes[1, 1].set_title('Custom CNN: Recall', fontsize=14, fontweight='bold')
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
        """Evaluate with memory-efficient batch processing"""
        print("\n" + "=" * 80)
        print("🧪 EVALUATING ON TEST SET")
        print("=" * 80)

        save_dir = Path(save_dir)

        # Evaluate
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

        # Get predictions in batches
        print("\n🔮 Generating predictions...")
        self.test_generator.reset()
        
        y_pred = []
        y_true = []
        
        for i in range(len(self.test_generator)):
            batch_pred = self.model.model.predict(self.test_generator[i][0], verbose=0)
            y_pred.extend(np.argmax(batch_pred, axis=1))
            y_true.extend(np.argmax(self.test_generator[i][1], axis=1))
            
            if (i + 1) % 10 == 0:
                print(f"   Processed {i + 1}/{len(self.test_generator)} batches")
                gc.collect()  # Clean memory periodically

        y_pred = np.array(y_pred)
        y_true = np.array(y_true)

        # Classification report
        print("\n📋 Classification Report:")
        print("=" * 80)
        report = classification_report(y_true, y_pred, target_names=self.model.class_names, digits=4)
        print(report)

        report_path = save_dir / 'classification_report.txt'
        with open(report_path, 'w') as f:
            f.write("Custom CNN Classification Report\n" + "=" * 80 + "\n" + report)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.model.class_names,
                    yticklabels=self.model.class_names)
        plt.title('Confusion Matrix - Custom CNN', fontsize=16, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        cm_path = save_dir / 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"📊 Confusion matrix saved to: {cm_path}")
        plt.show()

        # Save results
        results = {
            'model': 'Custom CNN',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'test_accuracy': float(test_accuracy),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'test_f1_score': float(test_f1),
            'confusion_matrix': cm.tolist(),
            'class_names': self.model.class_names
        }

        results_path = save_dir / 'test_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Test results saved to: {results_path}")
        print("=" * 80)

        return test_accuracy


def main():
    print("\n" + "=" * 80)
    print("🧠 MEMORY-EFFICIENT CUSTOM CNN TRAINING")
    print("=" * 80)
    print("Optimizations:")
    print("  ✅ Reduced batch size (16 instead of 64)")
    print("  ✅ Mixed precision training")
    print("  ✅ Automatic memory cleanup")
    print("  ✅ Frequent checkpointing")
    print("  ✅ Resume capability")
    print("=" * 80)

    # Configuration
    DATA_DIR = '/content/drive/MyDrive/Project Brain tumer classification/data'
    SAVE_DIR = '/content/drive/MyDrive/Project Brain tumer classification/custom_cnn_results'

    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 16  # Reduced from 64
    EPOCHS = 30
    LEARNING_RATE = 0.001
    NUM_CLASSES = 10

    if not os.path.exists(DATA_DIR):
        print(f"\n❌ Error: Data directory not found!")
        return

    # Verify directories
    for subdir in ['train', 'val', 'test']:
        path = os.path.join(DATA_DIR, subdir)
        if not os.path.exists(path):
            print(f"❌ Missing: {path}")
            return
        print(f"✅ Found: {path}")

    # Create model
    print("\n🏗️  Building Custom CNN model...")
    cnn_model = CustomCNN(input_shape=(*IMAGE_SIZE, 3), num_classes=NUM_CLASSES)
    cnn_model.build_model()
    cnn_model.compile_model(learning_rate=LEARNING_RATE)

    # Create trainer
    trainer = MemoryEfficientTrainer(
        data_dir=DATA_DIR,
        model=cnn_model,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE
    )

    trainer.setup_data_generators()
    
    # Train
    history = trainer.train(epochs=EPOCHS, save_dir=SAVE_DIR)
    
    if history is not None:
        trainer.plot_training_history(save_dir=SAVE_DIR)
        trainer.evaluate_on_test(save_dir=SAVE_DIR)
    
    print("\n✅ TRAINING COMPLETE!")
    print(f"📂 Results saved to: {SAVE_DIR}/")


if __name__ == "__main__":
    main()
