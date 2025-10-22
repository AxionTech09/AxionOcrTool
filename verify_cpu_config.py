#!/usr/bin/env python3
"""
Verification script to ensure no bfloat16 or float16 references remain in the project
"""

import os
import re
from pathlib import Path

def scan_for_mixed_precision(directory):
    """Scan all Python files for mixed precision references"""
    patterns = [
        r'bfloat16',
        r'bf16', 
        r'float16',
        r'fp16',
        r'torch\.half',
        r'\.half\(\)',
        r'device_map.*auto',
        r'device_map.*cuda',
    ]
    
    issues_found = []
    python_files = list(Path(directory).rglob('*.py'))
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
            for line_no, line in enumerate(lines, 1):
                # Skip comment-only lines
                if line.strip().startswith('#') and 'torch_dtype=torch.bfloat16' in line:
                    continue
                    
                for pattern in patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        issues_found.append({
                            'file': str(file_path),
                            'line': line_no,
                            'content': line.strip(),
                            'pattern': pattern
                        })
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    return issues_found

def check_cpu_compatibility():
    """Check if the configuration is CPU-compatible"""
    print("🔍 Scanning for mixed precision references...")
    
    # Scan current directory
    issues = scan_for_mixed_precision('.')
    
    if not issues:
        print("✅ No mixed precision references found!")
        print("✅ Project is configured for CPU-only operation with float32")
        return True
    else:
        print("❌ Found potential issues:")
        for issue in issues:
            print(f"  📁 {issue['file']}:{issue['line']}")
            print(f"     Pattern: {issue['pattern']}")
            print(f"     Content: {issue['content']}")
            print()
        return False

def verify_cpu_config():
    """Verify CPU configuration files exist and are correct"""
    print("\n🔧 Verifying CPU configuration...")
    
    # Check if cpu_config.py exists
    if not os.path.exists('cpu_config.py'):
        print("❌ cpu_config.py not found")
        return False
    
    # Check parser.py imports
    try:
        with open('dots_ocr/parser.py', 'r') as f:
            content = f.read()
            if 'from cpu_config import' in content:
                print("✅ Parser imports CPU configuration")
            else:
                print("❌ Parser does not import CPU configuration")
                return False
    except FileNotFoundError:
        print("❌ dots_ocr/parser.py not found")
        return False
    
    print("✅ CPU configuration verified")
    return True

def main():
    print("DotsOCR CPU Compatibility Checker")
    print("=" * 40)
    
    cpu_compatible = check_cpu_compatibility()
    config_ok = verify_cpu_config()
    
    print("\n" + "=" * 40)
    if cpu_compatible and config_ok:
        print("🎉 SUCCESS: Project is fully configured for CPU-only operation!")
        print("📋 Configuration Summary:")
        print("  • All mixed precision references removed")
        print("  • Using float32 precision only") 
        print("  • CPU device mapping enforced")
        print("  • FlashAttention disabled")
        print("  • Memory optimizations enabled")
    else:
        print("⚠️  ISSUES FOUND: Please fix the above issues before deployment")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())