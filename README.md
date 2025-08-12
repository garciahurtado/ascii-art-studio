# ASCII Art Studio

A powerful Python application that converts images and videos into ASCII art using PyTorch. This project features both a desktop application and a web interface for generating ASCII art from various media sources.

![ASCII Art Example](web/static/images/ascii_art_example.png)
*Example of ASCII art generated from an image*

### Screenshots

[<img src="web/static/examples/web_player.png" width="300" style="margin: 10px;"/>](web/static/examples/web_player.png)
[<img src="web/static/examples/web_converter.png" width="300" style="margin: 10px;"/>](web/static/examples/web_converter.png)
*Click on any image to view full size*

## Features

- **Image to ASCII Conversion**: Convert any image into beautiful ASCII art
- **Video to ASCII Animation**: Transform videos into animated ASCII art
- **Multiple Character Sets**: Support for various character sets including C64 and Ubuntu Mono
- **Desktop Application**: Full-featured desktop player with real-time preview
- **Web Interface**: Easy-to-use web application for quick conversions
- **Neural Network Based**: Utilizes PyTorch for high-quality ASCII conversion
- **Color Support**: Preserve color information in the output
- **Customizable Output**: Adjustable character size, contrast, and brightness

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ascii_movie_pytorch.git
   cd ascii_movie_pytorch
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Desktop Application

Run the desktop player:
```bash
python desktop_player.py [video_file]
```

### Web Application

Start the web server:
```bash
cd web
python app.py
```

Then open your browser and navigate to `http://localhost:8080`

### Command Line Tool

Convert an image to ASCII art:
```bash
python console_player.py path/to/your/image.jpg
```

## Project Structure

- `bin/`: Command-line tools and utilities
- `datasets/`: Training datasets and character sets
- `lib/`: Core library code
- `models/`: Trained PyTorch models
- `training/`: Training scripts and utilities
- `web/`: Web application code
  - `static/`: Static files (CSS, JS, images)
  - `app.py`: Main web application
- `desktop_player.py`: Desktop ASCII video player
- `console_player.py`: Command-line ASCII image converter

## Dependencies

- Python 3.8+
- PyTorch
- OpenCV
- NumPy
- Flask (for web interface)
- PySimpleGUI (for desktop interface)
- MoviePy (for video processing)

## Examples

### Video Conversion
```bash
python desktop_player.py your_video.mp4
```

### Image Conversion
```bash
python console_player.py path/to/image.jpg
```

