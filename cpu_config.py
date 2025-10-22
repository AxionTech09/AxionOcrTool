"""
CPU-Optimized Configuration for DotsOCR

This configuration ensures the project runs efficiently on CPU-only VPS servers
without any GPU dependencies or mixed precision operations.
"""

import torch
import os

# Force CPU-only operation
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TORCH_USE_CPU_ONLY'] = '1'

# CPU-optimized model configuration
CPU_MODEL_CONFIG = {
    'torch_dtype': torch.float32,          # CPU-friendly precision
    'device_map': 'cpu',                   # Force CPU device mapping
    'attn_implementation': None,           # Disable flash attention
    'low_cpu_mem_usage': True,             # Enable CPU memory optimization
    'trust_remote_code': True,             # Required for custom models
}

# Alternative fallback configuration
FALLBACK_CONFIG = {
    'torch_dtype': torch.float32,
    'device_map': None,                    # No automatic device mapping
    'trust_remote_code': True,
    'low_cpu_mem_usage': False
}

# Inference settings optimized for CPU
CPU_INFERENCE_CONFIG = {
    'max_new_tokens': 16384,               # Reduced for memory efficiency
    'temperature': 0.1,
    'top_p': 1.0,
    'num_threads': 1,                      # Single thread for HF model
    'use_cache': True,                     # Enable caching for efficiency
}

def ensure_cpu_dtype(model):
    """
    Ensures all model parameters and buffers are in float32 on CPU
    """
    model = model.float().to("cpu")
    for param in model.parameters():
        param.data = param.data.to(torch.float32)
    for buffer in model.buffers():
        buffer.data = buffer.data.to(torch.float32)
    model.eval()
    return model

def convert_inputs_to_cpu_float32(inputs):
    """
    Converts all tensor inputs to float32 on CPU
    """
    def convert_tensor_dict(tensor_dict):
        converted = {}
        for key, value in tensor_dict.items():
            if isinstance(value, torch.Tensor):
                if value.dtype.is_floating_point:
                    converted[key] = value.to(torch.float32).to("cpu")
                else:
                    converted[key] = value.to("cpu")
            elif isinstance(value, dict):
                converted[key] = convert_tensor_dict(value)
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                converted[key] = [
                    t.to(torch.float32).to("cpu") if t.dtype.is_floating_point else t.to("cpu") 
                    for t in value
                ]
            else:
                converted[key] = value
        return converted
    
    return convert_tensor_dict(inputs)

def optimize_for_cpu():
    """
    Apply CPU-specific optimizations
    """
    torch.set_num_threads(torch.get_num_threads())
    torch.set_grad_enabled(False)
    print(f"PyTorch optimized for CPU with {torch.get_num_threads()} threads")
    print(f"Default tensor type: {torch.get_default_dtype()}")

# Auto-apply optimizations when module is imported
optimize_for_cpu()
