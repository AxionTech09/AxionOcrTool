# DotsOCR CPU-Only Deployment Guide

This guide explains how to deploy DotsOCR on a CPU-only VPS server without GPU dependencies.

## ⚡ Key Changes for CPU Operation

### 1. **Removed Mixed Precision**
- All `bfloat16` and `float16` references removed
- Using `float32` exclusively for CPU compatibility
- No more dtype mismatch errors

### 2. **CPU-Optimized Configuration**
- FlashAttention disabled (requires GPU)
- Device mapping forced to CPU
- Memory usage optimized for CPU operation

### 3. **Updated Dependencies**
- Removed `flash-attn` (GPU-only)
- Removed `accelerate` (not needed for CPU)
- Using CPU-only PyTorch

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run with HuggingFace Model (CPU)
```bash
python dots_ocr/parser.py "image.jpg" --use_hf --output "./output"
```

### 3. Verify CPU Configuration
```bash
python verify_cpu_config.py
```

## 📋 Configuration Details

### Model Loading
```python
# Before (GPU/Mixed Precision)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,  # ❌ Not CPU compatible
    device_map="auto"             # ❌ May use GPU
)

# After (CPU-Only)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,    # ✅ CPU compatible
    device_map="cpu"              # ✅ Force CPU
)
```

### Input Processing
```python
# Automatic conversion to float32
inputs = convert_inputs_to_cpu_float32(inputs)
```

## 🔧 Files Modified

### Core Changes
- `dots_ocr/parser.py` - Main parser with CPU optimizations
- `demo/demo_hf.py` - Demo updated for CPU operation
- `cpu_config.py` - CPU-specific configuration module

### New Files
- `cpu_config.py` - CPU optimization utilities
- `verify_cpu_config.py` - Verification script
- `requirements_cpu.txt` - CPU-only dependencies

## ⚠️ Important Notes

### Memory Usage
- CPU inference requires more memory than GPU
- Consider using smaller batch sizes
- Monitor memory usage with `htop` or similar

### Performance
- CPU inference is slower than GPU
- Single-threaded for HF model (`num_thread=1`)
- Use VLLM server for better performance if possible

### Threading
- HF model uses single thread to avoid conflicts
- PDF processing still uses multiple threads
- Adjust `--num_thread` parameter as needed

## 🐛 Troubleshooting

### Common Issues

1. **"RuntimeError: Input type and bias type should be the same"**
   - ✅ Fixed: All tensors converted to float32

2. **"CUDA not available"**
   - ✅ Fixed: Force CPU device mapping

3. **"flash_attention not found"**
   - ✅ Fixed: FlashAttention disabled

4. **Memory errors**
   - Reduce `max_new_tokens` parameter
   - Use smaller images
   - Increase system swap space

### Verification Commands
```bash
# Check configuration
python verify_cpu_config.py

# Test model loading
python test_hf_model.py

# Monitor memory usage
htop
```

## 📊 Performance Expectations

### CPU vs GPU Performance
- **GPU**: ~2-5 seconds per page
- **CPU**: ~30-60 seconds per page
- **Memory**: 4-8GB RAM recommended

### Optimization Tips
1. Use smaller images when possible
2. Reduce `max_new_tokens` for faster processing
3. Process single pages instead of large PDFs
4. Consider using VLLM server for better throughput

## 🔄 Migration from GPU Setup

If migrating from a GPU setup:

1. **Remove GPU dependencies**:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

2. **Update model loading**:
   - Replace all `device_map="auto"` with `device_map="cpu"`
   - Change `torch_dtype=torch.bfloat16` to `torch_dtype=torch.float32`

3. **Test the configuration**:
   ```bash
   python verify_cpu_config.py
   ```

## 📞 Support

If you encounter issues:
1. Run the verification script first
2. Check the troubleshooting section
3. Ensure all dependencies are CPU versions
4. Monitor memory usage during processing