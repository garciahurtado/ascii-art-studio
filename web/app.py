import os
import sys
import time
import traceback
import uuid

from cvtools.processing_pipeline import ProcessingPipeline
from logging_config import logger

import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from io import BytesIO
import base64
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory, render_template_string, send_file
from werkzeug.utils import secure_filename
import tempfile

from color.palette import Palette

# Create the Flask app here so that the decorators can be used
app = Flask(__name__, static_folder='static')

CONVERT_TO_BGR = False

# Import the ASCII converter and processing pipeline
from ascii.neural_ascii_converter_pytorch import NeuralAsciiConverterPytorch
from charset import Charset
from schemas import validate_conversion_data

# Initialize the converter with C64 charset
CHAR_WIDTH, CHAR_HEIGHT = 8, 8
CHARSET_NAME = 'c64.png'
MODEL_FILENAME = 'ascii_c64'
MODEL_VERSION = '28'
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
    """Handle image upload and return rendered ASCII art as a PNG image."""
    temp_path = None

    try:
        # Validate input using our schema
        result = validate_conversion_data(request.form, request.files)
        
        # Check if validation failed (returns a tuple with errors)
        if isinstance(result, tuple) and len(result) == 2 and 'errors' in result[0]:
            logger.error(f"Validation failed: {result[0]}")
            return jsonify(result[0]), result[1]
        
        # If we get here, the data is valid
        form_data = result[0] if isinstance(result, tuple) and len(result) > 0 else result

        logger.info(f"convert_image form_data: {form_data}")
        
        upload = form_data['image']
        brightness = form_data['brightness']
        contrast = form_data['contrast']
        char_cols = form_data['char_cols']
        char_rows = form_data['char_rows']

        if not upload or upload.filename == '':
            error_msg = 'No file was uploaded'
            logger.error(error_msg)
            return jsonify({'error': error_msg, 'field': 'image'}), 400

        if not upload.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            error_msg = 'Unsupported file type. Please upload a PNG or JPG image.'
            logger.error(error_msg)
            return jsonify({'error': error_msg, 'field': 'image'}), 400

        # Create a temporary file to store the upload
        temp_path = os.path.join(tempfile.gettempdir(), secure_filename(upload.filename))

        upload.save(temp_path)

        # Verify the file was saved and is not empty
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            error_msg = 'Uploaded file is empty or could not be saved'
            logger.error(error_msg)
            return jsonify({'error': error_msg, 'field': 'image'}), 400

        logger.info(f"Processing image: {upload.filename} ({os.path.getsize(temp_path)} bytes)")

        # Initialize palette and converter
        palette = init_palette()
        if not palette:
            error_msg = 'Failed to initialize color palette'
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 500

        logger.info(f"Using palette: {palette}")
        logger.info(f"Number of colors in palette: {len(palette.colors) if palette else 0}")

        converter = init_converter()
        if not converter:
            error_msg = 'Failed to initialize ASCII converter'
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 500

        charset = converter.charset

        # Read and verify the image
        img = cv2.imread(temp_path)

        if img is None:
            error_msg = f'Failed to read image file: {upload.filename}'
            logger.error(error_msg)
            return jsonify({'error': error_msg, 'field': 'image'}), 400

        width = char_cols * charset.char_width
        height = char_rows * charset.char_height

        # Slightly crop the image dimensions if they are not an exact multiple of the character size
        height = int(height - (height % charset.char_height))
        width = int(width - (width % charset.char_width))

        # Process the image
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

        logger.info(f"Starting image processing: {width}x{height}, brightness={brightness}, contrast={contrast}")
        start_time = time.time()
        final_img = pipeline.run(img)

        # Save the contrast image, for debugging. Create a GUID type filename
        guid = str(uuid.uuid4())
        contrast_img_path = os.path.join(tempfile.gettempdir(), f'{guid}-contrast.png')
        cv2.imwrite(contrast_img_path, pipeline.contrast_img)
        logger.info(f"Saved contrast image to: {contrast_img_path}")

        # Convert back to RGB for web display
        final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)

        # Convert the color image to bytes and send it back as an HTTP response
        img_pil = Image.fromarray(final_img)
        img_io = BytesIO()
        img_pil.save(img_io, 'PNG')
        img_io.seek(0)

        elapsed_time = time.time() - start_time
        logger.info(f"Image processing completed successfully in {elapsed_time:.2f} seconds")
        return send_file(
            img_io,
            mimetype='image/png',
            as_attachment=False
        )

    except ValueError as ve:
        error_msg = f'Invalid parameter value: {str(ve)}'
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return jsonify({'error': error_msg, 'type': 'value_error'}), 400

    except KeyError as ke:
        error_msg = f'Missing required parameter: {str(ke)}'
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return jsonify({'error': error_msg, 'type': 'missing_field', 'field': str(ke)}), 400

    except Exception as e:
        error_msg = f'An unexpected error occurred: {str(e)}'
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return jsonify({
            'error': 'An internal server error occurred',
            'details': str(e),
            'type': 'unexpected_error'
        }), 500

    finally:
        # Clean up the temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.error(f"Non Fatal Error deleting temp file {temp_path}: str(e)")

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


def get_pipeline(converter, height, width, char_height, char_width, palette, brightness=0, contrast=1):
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
    pipeline = ProcessingPipeline(brightness=brightness, contrast=contrast)

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
