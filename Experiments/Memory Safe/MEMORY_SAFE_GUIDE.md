# Memory-Safe Training Guide
## For Colab Kernel Crash Prevention

If your Colab kernel is crashing during training, use these memory-optimized scripts.

---

## 🚨 Why Crashes Happen

Common causes of Colab crashes:
1. **Large batch sizes** (64, 32) consume too much GPU memory
2. **No memory cleanup** between epochs
3. **Multiple models** loaded simultaneously
4. **Large datasets** (9,000+ images)

---

## ✅ Memory-Safe Scripts

### **Scripts with Anti-Crash Protections:**

1. **`train_custom_cnn_memory_safe.py`**
   - Batch size: 16 (vs 64 original)
   - Mixed precision training
   - Auto memory cleanup
   - Resume capability

2. **`train_efficientnet_memory_safe.py`**
   - Batch size: 8 (vs 32 original)
   - Simplified, compact code
   - Memory cleanup callbacks

3. **`train_inception_memory_safe.py`**
   - Batch size: 8 (vs 64 original)
   - 299×299 image handling optimized
   - Memory cleanup callbacks

---

## 🚀 How to Use

### **Step 1: Check GPU Memory**
```python
from tensorflow.keras import backend as K

# Check GPU availability
import tensorflow as tf
print("GPU devices:", tf.config.list_physical_devices('GPU'))

# Enable memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("✅ GPU memory growth enabled")
```

### **Step 2: Run Memory-Safe Script**
```python
# Choose one model:

# Option 1: Custom CNN (safest, smallest model)
%run train_custom_cnn_memory_safe.py

# Option 2: EfficientNet-B4 (best accuracy, moderate memory)
%run train_efficientnet_memory_safe.py

# Option 3: Inception-v4 (good accuracy, needs more memory)
%run train_inception_memory_safe.py
```

### **Step 3: If Crash Occurs - Resume Training**
The Custom CNN script supports resume:
```python
# Just run again - it will ask if you want to resume
%run train_custom_cnn_memory_safe.py

# Select 'y' when prompted:
# "Resume from checkpoint? (y/n): y"
```

---

## ⚙️ Memory Optimizations Applied

| Optimization | Original | Memory-Safe | Benefit |
|--------------|----------|-------------|---------|
| **Batch Size** | 64/32 | 16/8 | 4-8x less GPU memory |
| **Mixed Precision** | No | Yes | 2x less memory, faster |
| **Memory Cleanup** | No | After each epoch | Prevents accumulation |
| **Checkpointing** | Best only | Every epoch | Resume after crash |
| **Batch Processing** | Full test set | Batch-by-batch | Safer evaluation |

---

## 📊 Trade-offs

| Aspect | Original | Memory-Safe |
|--------|----------|-------------|
| **Training Speed** | Faster | ~2x slower |
| **Memory Usage** | High | Low |
| **Crash Risk** | High | Very Low |
| **Final Accuracy** | Same | Same |
| **GPU Utilization** | 95%+ | 60-80% |

**Bottom line:** Takes longer but won't crash!

---

## 💡 Additional Memory-Saving Tips

### **1. Use Runtime Type: GPU**
```
Runtime → Change runtime type → GPU (T4)
```

### **2. Close Other Tabs/Notebooks**
- Only keep one Colab notebook open
- Close Drive files/folders

### **3. Restart Runtime Before Training**
```python
# In Colab:
Runtime → Restart runtime
```

### **4. Monitor GPU Memory**
```python
!nvidia-smi
```

### **5. Process One Model at a Time**
Don't train multiple models in sequence without restarting.

```python
# BAD (will crash):
%run train_custom_cnn_memory_safe.py
%run train_efficientnet_memory_safe.py  # Crash here!

# GOOD:
%run train_custom_cnn_memory_safe.py
# Then: Runtime → Restart runtime
%run train_efficientnet_memory_safe.py
```

---

## 🔧 If Still Crashing

### **Reduce Batch Size Further**

Edit the script and change:
```python
# In train_custom_cnn_memory_safe.py
BATCH_SIZE = 8  # Change from 16 to 8

# In train_efficientnet_memory_safe.py
BATCH_SIZE = 4  # Change from 8 to 4

# In train_inception_memory_safe.py  
BATCH_SIZE = 4  # Change from 8 to 4
```

### **Use Colab Pro**
- More RAM (25GB vs 12GB)
- Better GPUs (V100, A100)
- Longer runtime

### **Train Locally**
If you have a GPU:
```bash
pip install tensorflow
python train_custom_cnn_memory_safe.py
```

---

## 📈 Expected Training Times (with GPU)

| Model | Original | Memory-Safe |
|-------|----------|-------------|
| Custom CNN | 20 min | 40 min |
| EfficientNet-B4 | 25 min | 50 min |
| Inception-v4 | 30 min | 60 min |

**Worth it to avoid crashes!** ✅

---

## ✅ Recommended Workflow

```python
# 1. Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Enable GPU memory growth
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# 3. Train ONE model
%run train_efficientnet_memory_safe.py  # Best choice

# 4. If need to train another model, restart runtime first
# Runtime → Restart runtime

# 5. Then train next model
%run train_custom_cnn_memory_safe.py
```

---

## 🎯 Which Script to Use?

### **Start with:** `train_efficientnet_memory_safe.py`
- Best accuracy (99.7%)
- Moderate memory usage
- Good balance

### **If that crashes:** `train_custom_cnn_memory_safe.py`
- Smallest model
- Lowest memory
- Most stable

### **If you have resources:** `train_inception_memory_safe.py`
- High accuracy (99.3%)
- Needs more memory
- Larger images (299×299)

---

## 📝 Checklist Before Training

- [ ] GPU runtime enabled
- [ ] Other notebooks closed
- [ ] Runtime restarted
- [ ] Google Drive mounted
- [ ] Data directory verified
- [ ] GPU memory growth enabled
- [ ] Only training ONE model

**Ready to train without crashes!** 🚀

---

## 🆘 Troubleshooting

### Crash: "ResourceExhaustedError"
**Solution:** Reduce batch size further (see above)

### Crash: "Out of memory"  
**Solution:** Restart runtime, close other tabs

### Crash: "CUDA error"
**Solution:** 
```python
# Restart runtime, then:
import tensorflow as tf
tf.config.experimental.set_memory_growth(
    tf.config.list_physical_devices('GPU')[0], True
)
```

### Training very slow
**Solution:** This is normal with smaller batches. Be patient!

### Can't resume training
**Solution:** Only Custom CNN supports resume. Others train from scratch.

---

**The memory-safe scripts will prevent crashes at the cost of longer training time!**
