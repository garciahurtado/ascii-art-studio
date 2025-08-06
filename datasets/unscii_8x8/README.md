# Unifont 8x8 Dataset

## Overview
This dataset contains ASCII art representations of the Unifont 8x8 character set. Each character is rendered as an 8x8 binary image patch, designed for training and evaluating multilingual character recognition models.

## Dataset Structure
```
unscii_8x8/
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
- **Character Set**: Basic multilingual plane (BMP) characters

## Dataset Details
- **Source**: GNU Unifont 8x8
- **Character Coverage**: Basic Multilingual Plane (BMP)
- **Splits**: Training/Validation/Test

## Usage
```python
from datasets.unscii_8x8 import Unifont8x8

dataset = Unifont8x8(train=True)
print(f"Dataset size: {len(dataset)}")
image, label = dataset[0]  # Returns (8, 8) tensor and label
```

## Processing Pipeline
1. Unifont 8x8 font is loaded
2. Each character is rendered as an 8x8 grayscale image
3. Images are normalized to [0, 1] range
4. Dataset is split into training and test sets

## License
[Specify your license here]

## References
- [GNU Unifont](http://unifoundry.com/unifont/)
- [Project Repository](https://github.com/yourusername/ascii_movie_pytorch)
