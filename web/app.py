import os
import sys
import traceback
import logging

import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from io import BytesIO
import base64
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory, render_template_string, send_file

# Create the Flask app here so that the decorators can be used
app = Flask(__name__, static_folder='static')

CONVERT_TO_BGR = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Import the ASCII converter and processing pipeline
from ascii.neural_ascii_converter_pytorch import NeuralAsciiConverterPytorch
from charset import Charset
from cvtools.processing_pipeline_color import ProcessingPipelineColor

# Initialize the converter with C64 charset
CHAR_WIDTH, CHAR_HEIGHT = 8, 8
CHARSET_NAME = 'c64.png'
MODEL_FILENAME = 'ascii_c64-Mar17_21-33-46'
MODEL_CHARSET = 'ascii_c64'

# CHARSET_NAME = 'amstrad-cpc.png'
# MODEL_FILENAME = 'AsciiAmstradCPC-Mar06_23-14-37'
# MODEL_CHARSET = 'AsciiAmstradCPC'

def init_converter():
    """Initialize the ASCII converter with C64 settings.
    
    Returns:
        tuple: (converter, pipeline_ascii) or (None, None, None) if initialization fails
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


        # Initialize the processing pipeline
        pipeline_ascii = ProcessingPipelineColor()
        pipeline_ascii.converter = converter

        return converter, pipeline_ascii
        
    except Exception as e:
        logger.error(f"Failed to initialize C64 converter: {e}")
        return None, None, None

# Initialize the converter and pipelines
converter, pipeline_ascii = init_converter()

# Configuration
STATIC_ROOT = 'static'
TEMP_DIR = Path('temp')
TEMP_DIR.mkdir(exist_ok=True)

# Ensure the temp directory exists
os.makedirs(TEMP_DIR, exist_ok=True)

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
    global converter, pipeline_ascii, pipeline_color
    temp_path = None
    
    try:
        # Check if the post request has the file part
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
            
        upload = request.files['image']
        if upload.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Get character dimensions from form data with defaults
        try:
            char_cols = int(request.form.get('char_cols', 80))
            char_rows = int(request.form.get('char_rows', 40))
            # Ensure minimum values
            char_cols = max(10, min(200, char_cols))
            char_rows = max(10, min(200, char_rows))
        except (ValueError, TypeError):
            char_cols, char_rows = 80, 40  # Default values if parsing fails
        
        # Validate file extension
        filename = upload.filename.lower()
        if not (filename.endswith(('.png', '.jpg', '.jpeg'))):
            return jsonify({'error': 'Unsupported file format. Please upload a PNG or JPG image.'}), 400
        
        # Save the uploaded file temporarily
        temp_path = TEMP_DIR / f'upload_{os.urandom(8).hex()}{os.path.splitext(filename)[1]}'
        upload.save(str(temp_path))
        
        try:
            # Open and process the image
            with Image.open(temp_path) as img:
                # Convert to numpy array (BGR format expected by OpenCV)
                if CONVERT_TO_BGR:  # BGR:
                    img_np = np.array(img)[:, :, ::-1]  # RGB to BGR
                else:
                    img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                
                # Calculate target dimensions based on character grid
                char_size = 8  # Character block size
                target_width = char_cols * char_size
                target_height = char_rows * char_size
                
                # Set pipeline dimensions
                pipeline_ascii.img_width = target_width
                pipeline_ascii.img_height = target_height
                pipeline_ascii.char_width = char_size
                pipeline_ascii.char_height = char_size

                # Run the processing pipeline
                color_img = pipeline_ascii.run(img_np)

                # Convert back to RGB for web display
                color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
                
                # Convert the color image to bytes
                img_pil = Image.fromarray(color_img)
                img_io = BytesIO()
                img_pil.save(img_io, 'PNG')
                img_io.seek(0)
                
                # Return the image directly
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

# Initialize Flask app

def init_app():
    """Initialize the Flask application."""
    # Clean up any existing temp files on startup
    cleanup_temp_files()

    return app

if __name__ == '__main__':
    app = init_app()
    app.run(debug=True, host='0.0.0.0', port=8080)
