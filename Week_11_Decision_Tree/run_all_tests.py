#!/usr/bin/env python3
"""Test runner for Week 11 Decision Tree sample test cases"""

import sys
import subprocess
from pathlib import Path

def run_test(input_file, output_file, test_type):
    """Run a single test case and compare output"""
    with open(input_file, 'r') as f:
        input_data = f.read()
    
    with open(output_file, 'r') as f:
        expected = f.read().strip()
    
    # Run submission.py with input
    result = subprocess.run(
        ['python3', 'template/submission.py'],
        input=input_data,
        capture_output=True,
        text=True,
        cwd='/Users/amberathome/Documents/Education/UIUC/CS 412 Intro to Data Mining/Week_11_Decision_Tree'
    )
    
    actual = result.stdout.strip()
    
    # Compare output
    passed = actual == expected
    
    return passed, actual, expected

def main():
    base_path = Path('/Users/amberathome/Documents/Education/UIUC/CS 412 Intro to Data Mining/Week_11_Decision_Tree/sample_test_cases')
    
    test_categories = ['classification', 'split_info', 'tree_structure']
    
    all_passed = True
    
    for category in test_categories:
        print(f"\n{'='*70}")
        print(f"Testing: {category.upper()}")
        print(f"{'='*70}\n")
        
        category_path = base_path / category
        
        # Find all input files
        input_files = sorted(category_path.glob('input*.txt'))
        
        for input_file in input_files:
            # Get corresponding output file
            output_file = category_path / input_file.name.replace('input', 'output')
            
            if not output_file.exists():
                print(f"⚠️  {input_file.name}: No corresponding output file")
                continue
            
            try:
                passed, actual, expected = run_test(input_file, output_file, category)
                
                if passed:
                    print(f"✅ {input_file.name}: PASS")
                else:
                    print(f"❌ {input_file.name}: FAIL")
                    print(f"   Expected:\n{expected[:200]}")
                    print(f"   Got:\n{actual[:200]}")
                    all_passed = False
            
            except Exception as e:
                print(f"❌ {input_file.name}: ERROR - {e}")
                all_passed = False
    
    print(f"\n{'='*70}")
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print(f"{'='*70}\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
