"""
Test script to verify character ordering stability.
Tests that characters with similar coordinates always sort in the same order.
"""
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alpr.config import ALPRConfig
from alpr.YOLO.char_organizer import CharacterOrganizer


def create_test_characters_similar_x():
    """Create test characters with very similar X coordinates."""
    # Characters with X coords that differ by < 0.01 pixels
    return [
        {"box": [100.000, 10, 110, 30], "class": "A", "confidence": 0.9},
        {"box": [100.001, 10, 110, 30], "class": "B", "confidence": 0.9},
        {"box": [100.002, 10, 110, 30], "class": "C", "confidence": 0.9},
        {"box": [150.000, 10, 160, 30], "class": "1", "confidence": 0.9},
        {"box": [150.001, 10, 160, 30], "class": "2", "confidence": 0.9},
        {"box": [150.002, 10, 160, 30], "class": "3", "confidence": 0.9},
    ]


def create_test_characters_identical_x():
    """Create test characters with identical X coordinates."""
    return [
        {"box": [100, 10, 110, 20], "class": "A", "confidence": 0.9},
        {"box": [100, 25, 110, 35], "class": "B", "confidence": 0.9},
        {"box": [100, 40, 110, 50], "class": "C", "confidence": 0.9},
        {"box": [150, 10, 160, 20], "class": "1", "confidence": 0.9},
        {"box": [150, 25, 160, 35], "class": "2", "confidence": 0.9},
    ]


def create_test_characters_two_char():
    """Create test with exactly 2 characters (special case)."""
    return [
        {"box": [100.0001, 10, 110, 30], "class": "A", "confidence": 0.9},
        {"box": [100.0000, 10, 110, 30], "class": "B", "confidence": 0.9},
    ]


def create_test_characters_multiline():
    """Create test characters for multi-line plates."""
    return [
        # Top line
        {"box": [10, 10, 20, 30], "class": "A", "confidence": 0.9},
        {"box": [25, 10, 35, 30], "class": "B", "confidence": 0.9},
        {"box": [40, 10, 50, 30], "class": "C", "confidence": 0.9},
        # Bottom line
        {"box": [10, 50, 20, 70], "class": "1", "confidence": 0.9},
        {"box": [25, 50, 35, 70], "class": "2", "confidence": 0.9},
        {"box": [40, 50, 50, 70], "class": "3", "confidence": 0.9},
    ]


def get_character_sequence(organized_chars):
    """Extract the sequence of character classes."""
    return ''.join([char.get('class', '?') for char in organized_chars])


def test_stability(test_name, characters, iterations=1000):
    """Test that character ordering is stable across multiple runs."""
    print(f"\n{'='*60}")
    print(f"Testing: {test_name}")
    print(f"{'='*60}")
    
    # Create minimal config
    config = ALPRConfig(
        save_debug_images=False,
        enable_state_detection=False,
        enable_vehicle_detection=False
    )
    
    organizer = CharacterOrganizer(config)
    
    sequences = []
    for i in range(iterations):
        # Create fresh copy each iteration to avoid any object mutation
        chars_copy = [char.copy() for char in characters]
        organized = organizer.organize_characters(chars_copy)
        sequence = get_character_sequence(organized)
        sequences.append(sequence)
    
    # Check if all sequences are identical
    unique_sequences = set(sequences)
    
    if len(unique_sequences) == 1:
        print(f"✅ PASS: All {iterations} iterations produced identical order")
        print(f"   Result: {sequences[0]}")
        return True
    else:
        print(f"❌ FAIL: Found {len(unique_sequences)} different orderings")
        for seq in unique_sequences:
            count = sequences.count(seq)
            print(f"   {seq}: {count} times ({count/iterations*100:.1f}%)")
        return False


def main():
    """Run all stability tests."""
    print("\n" + "="*60)
    print("CHARACTER ORDERING STABILITY TEST SUITE")
    print("="*60)
    
    tests = [
        ("Similar X coordinates (floating-point precision)", create_test_characters_similar_x()),
        ("Identical X coordinates", create_test_characters_identical_x()),
        ("Two characters (edge case)", create_test_characters_two_char()),
        ("Multi-line plate", create_test_characters_multiline()),
    ]
    
    results = []
    for test_name, characters in tests:
        passed = test_stability(test_name, characters, iterations=1000)
        results.append((test_name, passed))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    for test_name, passed_flag in results:
        status = "✅ PASS" if passed_flag else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Character ordering is stable.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Character ordering may be unstable.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
