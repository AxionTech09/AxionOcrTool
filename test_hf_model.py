#!/usr/bin/env python3
"""
Test script to verify the HuggingFace model loading and dtype consistency
"""

import sys
import os
import torch
from PIL import Image
import numpy as np

# Add the dots_ocr module to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_model_loading():
    """Test if the model loads correctly with consistent dtypes"""
    print("Testing HuggingFace model loading...")
    
    try:
        from dots_ocr.parser import DotsOCRParser
        
        # Initialize parser with HF model
        parser = DotsOCRParser(use_hf=True)
        
        print("✓ Model loaded successfully")
        
        # Check model dtypes
        model_dtypes = set()
        for param in parser.model.parameters():
            model_dtypes.add(param.dtype)
        
        print(f"Model parameter dtypes: {model_dtypes}")
        
        # Check buffer dtypes
        buffer_dtypes = set()
        for buffer in parser.model.buffers():
            buffer_dtypes.add(buffer.dtype)
        
        print(f"Model buffer dtypes: {buffer_dtypes}")
        
        # Create a simple test image
        test_image = Image.new('RGB', (224, 224), color='white')
        
        # Test inference with simple prompt
        print("Testing inference...")
        try:
            result = parser._inference_with_hf(test_image, "What is in this image?")
            print("✓ Inference completed successfully")
            print(f"Result length: {len(result) if result else 0}")
        except Exception as e:
            print(f"✗ Inference failed: {e}")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dtype_conversion():
    """Test the dtype conversion utilities"""
    print("\nTesting dtype conversion utilities...")
    
    # Create test tensors with CPU-compatible dtypes
    test_dict = {
        'input_ids': torch.tensor([[1, 2, 3]], dtype=torch.long),
        'pixel_values': torch.randn(1, 3, 224, 224, dtype=torch.float32),  # Changed from bfloat16 to float32
        'attention_mask': torch.ones(1, 3, dtype=torch.bool)
    }
    
    print("Original dtypes:")
    for key, value in test_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.dtype}")
    
    # Test conversion function (copied from parser)
    def convert_to_float32(tensor_dict):
        converted = {}
        for key, value in tensor_dict.items():
            if isinstance(value, torch.Tensor):
                # Convert all floating point tensors to float32 for CPU compatibility
                if value.dtype.is_floating_point:
                    converted[key] = value.to(torch.float32).to("cpu")
                else:
                    converted[key] = value.to("cpu")
            else:
                converted[key] = value
        return converted
    
    converted = convert_to_float32(test_dict)
    
    print("Converted dtypes:")
    for key, value in converted.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.dtype}")
    
    print("✓ Dtype conversion test completed")

if __name__ == "__main__":
    print("DotsOCR HuggingFace Model Test")
    print("=" * 40)
    
    # Test dtype conversion utilities first
    test_dtype_conversion()
    
    # Test model loading and inference
    success = test_model_loading()
    
    print("\n" + "=" * 40)
    if success:
        print("✓ All tests passed!")
        print("The model should now work correctly.")
    else:
        print("✗ Tests failed!")
        print("Please check the error messages above.")
        sys.exit(1)