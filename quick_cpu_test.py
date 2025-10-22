#!/usr/bin/env python3
"""
Quick CPU compatibility test
"""

def test_cpu_imports():
    """Test that all imports work with CPU configuration"""
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        
        # Check if CUDA is available but ensure we're using CPU
        print(f"✓ CUDA available: {torch.cuda.is_available()} (using CPU anyway)")
        print(f"✓ CPU threads: {torch.get_num_threads()}")
        
        # Test tensor operations
        x = torch.randn(3, 3, dtype=torch.float32)
        y = torch.matmul(x, x.t())
        print(f"✓ CPU tensor operations working: {y.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_model_config():
    """Test model configuration"""
    try:
        from cpu_config import CPU_MODEL_CONFIG, ensure_cpu_dtype, convert_inputs_to_cpu_float32
        print("✓ CPU configuration module imported successfully")
        print(f"✓ Model config: {CPU_MODEL_CONFIG}")
        
        # Test input conversion
        import torch
        test_inputs = {
            'input_ids': torch.tensor([[1, 2, 3]], dtype=torch.long),
            'pixel_values': torch.randn(1, 3, 224, 224, dtype=torch.float32)
        }
        
        converted = convert_inputs_to_cpu_float32(test_inputs)
        print("✓ Input conversion working")
        
        for key, value in converted.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.dtype} on {value.device}")
        
        return True
    except Exception as e:
        print(f"❌ Model config test failed: {e}")
        return False

def main():
    print("DotsOCR CPU Compatibility Quick Test")
    print("=" * 40)
    
    test1 = test_cpu_imports()
    test2 = test_model_config()
    
    print("\n" + "=" * 40)
    if test1 and test2:
        print("🎉 All tests passed! CPU configuration is ready.")
    else:
        print("❌ Some tests failed. Check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())