# Ubuntu Mono Dataset

## Overview
This dataset contains ASCII art representations of the Ubuntu Mono typeface. Each character is rendered as an 8x8 binary image patch, ideal for training monospace font recognition models.

## Dataset Structure
```
ubuntu_mono/
├── processed/       # Preprocessed character data in CSV format
└── reports/         # Analysis and statistics
```

## Data Format
- **File Format**: Tab-separated CSV files
- **Row Format**:
  - Column 1: Character label (integer)
  - Columns 2-65: 64 binary pixel values (8x8) in row-major order
- **Pixel Values**: 0 (black) or 1 (white)
- **Character Size**: 8x8 pixels
- **Character Set**: ASCII printable characters (32-126)

## Dataset Details
- **Source**: Ubuntu Mono typeface
- **Total Characters**: 95 (printable ASCII)
- **Splits**: Training/Validation/Test

## Usage
```python
from datasets.ubuntu_mono import UbuntuMono

dataset = UbuntuMono(train=True)
print(f"Dataset size: {len(dataset)}")
image, label = dataset[0]  # Returns (8, 8) tensor and label
```

## Processing Pipeline
1. Ubuntu Mono font is loaded
2. Each character is rendered as an 8x8 grayscale image
3. Images are normalized to [0, 1] range
4. Dataset is split into training and test sets

## License
[Specify your license here]

## References
- [Ubuntu Font License](https://ubuntu.com/legal/font-licence)
- [Project Repository](https://github.com/yourusername/ascii_movie_pytorch)
