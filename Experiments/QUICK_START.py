# Quick Start - Train All Models (No Augmentation Version)
# For pre-augmented data at: /content/drive/MyDrive/Project Brain tumer classification/data

print("""
================================================================================
🧠 BRAIN TUMOR CLASSIFICATION - TRAINING ALL MODELS
================================================================================

This will train all three models on your PRE-AUGMENTED data:
1. Custom CNN (96.5% expected)
2. EfficientNet-B4 (99.7% expected) ⭐ BEST
3. Inception-v4 (99.3% expected)

Total time: ~60-75 minutes with GPU
Results will be saved to your Google Drive

================================================================================
""")

import os

# Check data directory
data_dir = '/content/drive/MyDrive/Project Brain tumer classification/data'
if not os.path.exists(data_dir):
    print("❌ ERROR: Data directory not found!")
    print(f"   Expected: {data_dir}")
    print("\nPlease check:")
    print("1. Google Drive is mounted")
    print("2. Train/val/test split is complete")
else:
    print("✅ Data directory found!")
    
    # Ask which models to train
    print("\n" + "=" * 80)
    print("SELECT MODELS TO TRAIN:")
    print("=" * 80)
    print("1. All three models (recommended)")
    print("2. EfficientNet-B4 only (fastest, best accuracy)")
    print("3. Custom CNN only")
    print("4. Inception-v4 only")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    print("\n" + "=" * 80)
    
    if choice == '1':
        print("Training all three models...")
        print("=" * 80)
        print("\n🔵 STEP 1/3: Training Custom CNN...")
        exec(open('train_custom_cnn_no_aug.py').read())
        
        print("\n🟢 STEP 2/3: Training EfficientNet-B4...")
        exec(open('train_efficientnet_b4_no_aug.py').read())
        
        print("\n🟣 STEP 3/3: Training Inception-v4...")
        exec(open('train_inception_v4_no_aug.py').read())
        
    elif choice == '2':
        print("Training EfficientNet-B4 (Best Model)...")
        print("=" * 80)
        exec(open('train_efficientnet_b4_no_aug.py').read())
        
    elif choice == '3':
        print("Training Custom CNN...")
        print("=" * 80)
        exec(open('train_custom_cnn_no_aug.py').read())
        
    elif choice == '4':
        print("Training Inception-v4...")
        print("=" * 80)
        exec(open('train_inception_v4_no_aug.py').read())
        
    else:
        print("❌ Invalid choice. Please run again and select 1-4.")
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    print("\n📂 Results saved to Google Drive:")
    print("   /content/drive/MyDrive/Project Brain tumer classification/")
    print("=" * 80)
