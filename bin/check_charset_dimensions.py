#!/usr/bin/env python3
"""
Analyze character set image dimensions and calculate character grid information.

This script examines all PNG files in the character set directory and calculates
the number of characters based on the image dimensions and character size.
Character size is inferred from filenames (e.g., _8x16.png).

The ultimate goal is to generate JSON files for each character set to be used as metadata, including the number of
characters, character width and height.

Usage:
    python check_charset.py path/to/charset.png  # Analyze specific file
    python check_charset.py --all                # Analyze all PNG files in the directory
"""

import os
import re
import argparse
from PIL import Image
import json

def get_charset_info(image_path):
    """
    Analyze character set image and return dimension information.
    
    Args:
        image_path: Path to the character set image file
        
    Returns:
        Dictionary containing image and character dimension information
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            
            # Extract character dimensions from filename (e.g., _8x16.png)
            match = re.search(r'_(\d+)x(\d+)\.(png|json)$', image_path, re.IGNORECASE)
            if match:
                char_width = int(match.group(1))
                char_height = int(match.group(2))
            else:
                # Default to 8x8 if no dimensions in filename
                char_width = 8
                char_height = 8
            
            # Calculate character grid
            chars_per_row = width // char_width
            rows = height // char_height
            total_chars = chars_per_row * rows
            
            return {
                'path': image_path,
                'width': width,
                'height': height,
                'char_width': char_width,
                'char_height': char_height,
                'chars_per_row': chars_per_row,
                'rows': rows,
                'total_chars': total_chars,
                'valid': (width % char_width == 0) and (height % char_height == 0)
            }
    except Exception as e:
        return {
            'path': image_path,
            'error': str(e),
            'valid': False
        }

def format_dimensions(info):
    """Format dimension information for console output."""
    if 'error' in info:
        return f"Error: {info['error']}"
    
    valid_str = "✓" if info['valid'] else "✗ (dimensions not divisible by character size)"
    
    return f"""
File: {os.path.basename(info['path'])}
Image dimensions: {info['width']}x{info['height']} pixels
Character size: {info['char_width']}x{info['char_height']} pixels
Character grid: {info['chars_per_row']}x{info['rows']}
Total characters: {info['total_chars']}
Valid: {valid_str}
"""

def analyze_charset(image_path):
    """Analyze a single character set and print results."""
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return False
    
    info = get_charset_info(image_path)
    print(format_dimensions(info))
    
    if not info.get('valid', False):
        print("JSON would contain:")
        print(json.dumps({
            'name': os.path.splitext(os.path.basename(image_path))[0],
            'char_width': info['char_width'],
            'char_height': info['char_height'],
            'inverted_included': False,
            'num_chars': info['total_chars']
        }, indent=4))
    
    return info.get('valid', False)

def find_png_files(directory):
    """Find all PNG files in the specified directory."""
    png_files = []
    for filename in os.listdir(directory):
        if filename.lower().endswith('.png'):
            png_files.append(os.path.join(directory, filename))
    return sorted(png_files)

def main():
    parser = argparse.ArgumentParser(description='Analyze character set dimensions.')
    parser.add_argument('charset', nargs='?', help='Path to character set image file')
    parser.add_argument('--all', action='store_true', help='Process all PNG files in the directory')
    args = parser.parse_args()
    
    charset_dir = os.path.join('lib', 'charset', 'res', 'charsets')
    
    if args.all:
        print(f"Analyzing all PNG files in {charset_dir}")
        print("=" * 60)
        
        png_files = find_png_files(charset_dir)
        if not png_files:
            print("No PNG files found in the directory.")
            return
            
        for i, png_file in enumerate(png_files, 1):
            print(f"\n[{i}/{len(png_files)}] Analyzing {os.path.basename(png_file)}")
            print("-" * 60)
            analyze_charset(png_file)
    elif args.charset:
        analyze_charset(args.charset)
    else:
        parser.print_help()
        print("\nExamples:")
        print(f"  {os.path.basename(__file__)} lib/charset/res/charsets/unscii_8x16.png")
        print(f"  {os.path.basename(__file__)} --all")

if __name__ == "__main__":
    main()
