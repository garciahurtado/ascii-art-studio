# ASCII Amstrad CPC Dataset

## Overview
This dataset contains ASCII art representations of the Amstrad CPC character set. Each character is rendered as an 8x8 binary image patch, designed for training and evaluating character recognition models.

## Dataset Structure
```
ascii_amstrad_cpc/
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
- **Character Set**: 256 characters (0-255)

## Dataset Details
- **Source**: Amstrad CPC character set
- **Total Characters**: 256
- **Splits**: Training/Validation/Test

## Usage
```python
from datasets.ascii_amstrad_cpc import AsciiAmstradCPC

dataset = AsciiAmstradCPC(train=True)
print(f"Dataset size: {len(dataset)}")
image, label = dataset[0]  # Returns (8, 8) tensor and label
```

## Processing Pipeline
1. Original Amstrad CPC font is loaded
2. Each character is rendered as an 8x8 grayscale image
3. Images are normalized to [0, 1] range
4. Dataset is split into training and test sets

## License
[Specify your license here]

## References
- [Amstrad CPC Character Set](https://www.cpcwiki.eu/index.php/CPC_character_set)
- [Project Repository](https://github.com/yourusername/ascii_movie_pytorch)
