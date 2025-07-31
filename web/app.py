import os
import sys
import traceback
from my_logging import logger

import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from io import BytesIO
import base64
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory, render_template_string, send_file

from color.palette import Palette

# Create the Flask app here so that the decorators can be used
app = Flask(__name__, static_folder='static')

CONVERT_TO_BGR = False

# Import the ASCII converter and processing pipeline
from ascii.neural_ascii_converter_pytorch import NeuralAsciiConverterPytorch
from charset import Charset
from cvtools.processing_pipeline_color import ProcessingPipelineColor

# Initialize the converter with C64 charset
CHAR_WIDTH, CHAR_HEIGHT = 8, 8
CHARSET_NAME = 'c64.png'
MODEL_FILENAME = 'ascii_c64-Mar17_21-33-46'
MODEL_CHARSET = 'ascii_c64'
PALETTE_NAME = 'atari.png'

# CHARSET_NAME = 'amstrad-cpc.png'
# MODEL_FILENAME = 'AsciiAmstradCPC-Mar06_23-14-37'
# MODEL_CHARSET = 'AsciiAmstradCPC'

# Configuration
STATIC_ROOT = 'static'
TEMP_DIR = Path('temp')
TEMP_DIR.mkdir(exist_ok=True)

# Ensure the temp directory exists
os.makedirs(TEMP_DIR, exist_ok=True)

def init_converter():
    """Initialize the ASCII converter with C64 settings.
    """
    try:
        # Load the C64 charset
        charset = Charset(CHAR_WIDTH, CHAR_HEIGHT)
        charset.load(CHARSET_NAME, invert=False)
        NUM_LABELS = len(charset.chars)

        # Initialize the converter with C64 model
        converter = NeuralAsciiConverterPytorch(
            charset=charset,
            model_filename=MODEL_FILENAME,
            model_charset=MODEL_CHARSET,
            charsize=[CHAR_WIDTH, CHAR_HEIGHT],
            num_labels=NUM_LABELS
        )

        return converter

    except Exception as e:
        logger.error(f"Failed to initialize C64 converter: {e}")
        return None

def init_palette():
    # Initialize the palette
    palette = Palette(char_width=CHAR_WIDTH, char_height=CHAR_HEIGHT)
    palette.load(PALETTE_NAME)
    return palette

def cleanup_temp_files():
    """Remove all files in the temporary directory."""
    for file_path in TEMP_DIR.glob('*'):
        try:
            if file_path.is_file():
                file_path.unlink()
        except Exception as e:
            logger.error(f"Error deleting {file_path}: {e}")

def log_error(error, request=None):
    """Log detailed error information including stack trace and request context.
    
    Args:
        error: The exception that was raised
        request: The Flask request object (optional)
    """
    error_type = type(error).__name__
    error_message = str(error)
    
    # Get the full stack trace
    exc_type, exc_value, exc_traceback = sys.exc_info()
    stack_trace = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    
    # Get request information if available
    request_info = {}
    if request:
        request_info = {
            'method': request.method,
            'path': request.path,
            'query_string': request.query_string.decode('utf-8'),
            'content_type': request.content_type,
            'content_length': request.content_length,
            'remote_addr': request.remote_addr,
            'user_agent': request.user_agent.string
        }
    
    # Log the error with all details
    logger.error(
        "\n" + "="*80 + "\n"
        f"ERROR: {error_type}: {error_message}\n"
        f"Time: {datetime.utcnow().isoformat()}\n"
        f"Request: {request_info}\n"
        f"Traceback:\n{stack_trace}"
        "\n" + "="*80
    )

@app.route('/')
def index():
    """Serve the main index page."""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/converter')
def serve_converter():
    """Serve the image to ASCII art converter page."""
    return send_from_directory(app.static_folder, 'converter.html')


@app.route('/convert', methods=['POST'])
def convert_image():
    """Handle image upload and return rendered ASCII art as a PNG image.
    
    Returns:
        Response: The rendered ASCII art as a PNG image
    """
    temp_path = None
    
    try:
        # Check if the post request has the file part
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
            
        upload = request.files['image']
        if upload.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        """ Get conversion parameters from form data with defaults """
        # Get conversion parameters from form data with defaults
        try:
            char_cols = int(request.form.get('char_cols', 80))
            char_rows = int(request.form.get('char_rows', 40))

            # Ensure minimum values for character dimensions
            char_cols = max(10, min(200, char_cols))
            char_rows = max(10, min(200, char_rows))

            # Get brightness and contrast with defaults (0/0)
            brightness = int(request.form.get('brightness', 0))
            contrast = int(request.form.get('contrast', 0))

            # Ensure values are within valid range (0-100)
            brightness = max(0, min(100, brightness))
            contrast = max(0, min(100, contrast))

        except (ValueError, TypeError) as e:
            logger.warning(f"Error parsing form data: {e}")
            char_cols, char_rows = 80, 40  # Default values if parsing fails
            brightness, contrast = 0, 0  # Default values for brightness/contrast
        
        # Validate file extension
        filename = upload.filename.lower()
        if not (filename.endswith(('.png', '.jpg', '.jpeg'))):
            return jsonify({'error': 'Unsupported file format. Please upload a PNG or JPG image.'}), 400
        
        # Save the uploaded file temporarily
        temp_path = TEMP_DIR / f'upload_{os.urandom(8).hex()}{os.path.splitext(filename)[1]}'
        upload.save(str(temp_path))

        palette = init_palette()
        converter = init_converter()
        charset = converter.charset
        
        try:
            # Open and process the image
            img = cv2.imread(str(temp_path))
            height, width = img.shape[0], img.shape[1]

            width = char_cols * charset.char_width
            height = char_rows * charset.char_height

            # Adjust in case the image is not a multiple of the character size
            height = height - (height % charset.char_height)
            width = width - (width % charset.char_width)

            pipeline = get_pipeline(
                converter=converter,
                height=height,
                width=width,
                char_height=charset.char_height,
                char_width=charset.char_width,
                palette=palette,
                brightness=brightness,
                contrast=contrast
            )
            pipeline.palette = palette
            final_img = pipeline.run(img)

            # Convert back to RGB for web display
            final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)

            # Convert the color image to bytes
            img_pil = Image.fromarray(final_img)
            img_io = BytesIO()
            img_pil.save(img_io, 'PNG')
            img_io.seek(0)

            # Also return the image directly in the HTTP response
            return send_file(
                img_io,
                mimetype='image/png',
                as_attachment=False
            )
                
        except Exception as e:
            log_error(e, request)
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
            
    except Exception as e:
        log_error(e, request)
        return jsonify({'error': 'An unexpected error occurred'}), 500
        
    finally:
        # Clean up the temporary file
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception as e:
                logger.error(f"Error deleting temp file {temp_path}: {e}")

@app.route('/api/videos')
def list_videos():
    """
    List all available .cpeg video files in the charpeg directory.

    Returns:
        JSON response containing a list of video files with their names and paths
    """
    try:
        # Define the directory containing video files
        video_dir = os.path.join(app.static_folder, 'charpeg')

        # Get all .cpeg files in the directory
        video_files = []
        for filename in os.listdir(video_dir):
            if filename.endswith('.cpeg'):
                # Create a user-friendly name by removing the extension and any path
                name = os.path.splitext(filename)[0]
                name = name.replace('_', ' ').title()

                video_files.append({
                    'name': name,
                    'path': f'charpeg/{filename}'
                })

        return jsonify(video_files)

    except Exception as e:
        log_error(e, request)
        return jsonify({'error': str(e)}), 500

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files from the static directory."""
    return send_from_directory(app.static_folder, path)

@app.route('/temp/<filename>')
def serve_temp_file(filename):
    """Serve files from the temporary directory."""
    return send_from_directory(TEMP_DIR, filename)

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors."""
    return send_from_directory(app.static_folder, '404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors with detailed logging.
    
    Returns a generic error message to the client while logging the full details.
    """
    log_error(error, request)
    
    # Return a generic error message
    error_id = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    logger.error(f"Error ID: {error_id}")
    
    return jsonify({
        'error': 'An internal server error occurred',
        'error_id': error_id
    }), 500


def get_pipeline(converter, height, width, char_height, char_width, palette, brightness=0, contrast=0):
    """Create and configure a processing pipeline for ASCII art conversion.
    
    Args:
        converter: The ASCII converter to use
        height: Desired output height in pixels
        width: Desired output width in pixels
        char_height: Height of each character in pixels
        char_width: Width of each character in pixels
        palette: Color palette to use for the output
        brightness: Brightness adjustment (0-100, default 0)
        contrast: Contrast adjustment (0-100, mapped to 1.0-3.0, default 0 which maps to 1.0)
        
    Returns:
        Configured ProcessingPipelineColor instance
    """
    # Create pipeline with brightness and contrast settings
    pipeline = ProcessingPipelineColor(brightness=brightness, contrast=contrast)

    # Set other pipeline properties
    pipeline.converter = converter
    pipeline.img_width = width
    pipeline.img_height = height
    pipeline.char_height = char_height
    pipeline.char_width = char_width
    pipeline.palette = palette

    return pipeline

# Initialize Flask app
def init_app():
    """Initialize the Flask application."""
    # Clean up any existing temp files on startup
    cleanup_temp_files()

    return app

if __name__ == '__main__':
    app = init_app()
    app.run(debug=True, host='0.0.0.0', port=8080)
