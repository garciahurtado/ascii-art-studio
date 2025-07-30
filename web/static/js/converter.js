/**
 * Image to ASCII Art Converter
 *
 * Handles the client-side functionality for the image to ASCII art converter,
 * including drag-and-drop, file uploads, and displaying results.
 */

document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('imageInput');
    const uploadForm = document.getElementById('uploadForm');
    const loading = document.getElementById('loading');
    const result = document.getElementById('result');
    const preview = document.getElementById('preview');
    const copyBtn = document.getElementById('copyBtn');
    let file = null;

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    // Highlight drop zone when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    // Remove highlight when item leaves drop zone
    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    // Handle dropped files
    dropZone.addEventListener('drop', handleDrop, false);

    // Open file dialog when drop zone is clicked
    dropZone.addEventListener('click', () => fileInput.click());

    // Handle file input change
    fileInput.addEventListener('change', handleFileSelect, false);

    // Handle form submission
    uploadForm.addEventListener('submit', handleFormSubmit);

    // Handle copy to clipboard
    if (copyBtn) {
        copyBtn.addEventListener('click', copyToClipboard);
    }

    /**
     * Prevent default drag behaviors
     * @param {Event} e - The event object
     */
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    /**
     * Highlight the drop zone when an item is dragged over it
     */
    function highlight() {
        dropZone.classList.add('drag-over');
    }

    /**
     * Remove highlight from drop zone
     */
    function unhighlight() {
        dropZone.classList.remove('drag-over');
    }

    /**
     * Handle files dropped onto the drop zone
     * @param {DragEvent} e - The drop event
     */
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    /**
     * Handle file selection via file input
     */
    function handleFileSelect() {
        handleFiles(this.files);
    }

    /**
     * Process selected files and display the original image
     * @param {FileList} files - List of files to process
     */
    function handleFiles(files) {
        if (files.length > 0) {
            file = files[0];
            if (file && file.type.match('image.*')) {
                updateFileInfo(file.name);

                // Create a preview of the selected image
                const reader = new FileReader();
                reader.onload = function(e) {
                    // Create or update the preview container
                    let previewContainer = document.getElementById('imagePreviewContainer');
                    if (!previewContainer) {
                        previewContainer = document.createElement('div');
                        previewContainer.id = 'imagePreviewContainer';
                        previewContainer.className = 'preview-container';

                        // Insert the preview container after the file input
                        const uploadSection = document.querySelector('.upload-section');
                        if (uploadSection) {
                            uploadSection.parentNode.insertBefore(previewContainer, uploadSection.nextSibling);
                        }
                    }

                    // Clear previous preview
                    previewContainer.innerHTML = '';

                    // Create and append the image
                    const img = document.createElement('img');
                    img.src = e.target.result;
                    img.alt = 'Selected Image Preview';
                    img.className = 'original-image';
                    img.style.maxWidth = '100%';
                    img.style.maxHeight = '60vh';
                    img.style.display = 'block';
                    img.style.margin = '1rem auto';
                    img.style.borderRadius = '0.5rem';

                    previewContainer.appendChild(img);
                };
                reader.readAsDataURL(file);
            } else {
                showError('Please select a valid image file (PNG, JPG, JPEG)');
            }
        }
    }

    /**
     * Update UI to show selected file info
     * @param {string} fileName - Name of the selected file
     */
    function updateFileInfo(fileName) {
        const fileInfo = document.getElementById('fileInfo');
        if (!fileInfo) {
            const infoDiv = document.createElement('div');
            infoDiv.id = 'fileInfo';
            infoDiv.className = 'mt-2 text-sm text-gray-600';
            dropZone.after(infoDiv);
        }
        document.getElementById('fileInfo').textContent = `Selected: ${fileName}`;
    }

    /**
     * Ensure the result container is properly initialized
     * @returns {Object} Object containing result and preview elements
     */
    function initializeResultContainer() {
        let result = document.getElementById('result');
        let preview = document.getElementById('preview');

        // Create result container if it doesn't exist
        if (!result) {
            result = document.createElement('div');
            result.id = 'result';
            result.className = 'p-8 border-t border-gray-200';
            document.querySelector('.container').appendChild(result);
        }

        // Create preview container if it doesn't exist
        if (!preview) {
            preview = document.createElement('div');
            preview.id = 'preview';
            preview.className = 'preview-container';
            result.appendChild(preview);
        }

        return { result, preview };
    }

    /**
     * Handle form submission
     * @param {Event} e - The submit event
     */
    async function handleFormSubmit(e) {
        e.preventDefault();

        const fileInput = document.getElementById('imageInput');
        const uploadForm = document.getElementById('uploadForm');
        const result = document.getElementById('result');
        const charCols = document.getElementById('charCols');
        const charRows = document.getElementById('charRows');

        // Check if a file is selected
        if (!fileInput.files || fileInput.files.length === 0) {
            showError('Please select an image file first.');
            return;
        }

        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append('image', file);

        // Add columns and rows to form data if they exist
        if (charCols && charCols.value) {
            formData.append('char_cols', charCols.value);
        }
        if (charRows && charRows.value) {
            formData.append('char_rows', charRows.value);
        }
        // Show loading state and prepare result container
        uploadForm.parentElement.classList.add('hidden');
        if (result) {
            result.innerHTML = '';
            result.classList.remove('hidden');
        }

        // Display original image immediately
        const originalImageUrl = URL.createObjectURL(file);
        displayOriginalImage(originalImageUrl);

        try {
            const response = await fetch('/convert', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                let error;
                try {
                    const errorData = await response.json();
                    error = errorData.error || 'Conversion failed';
                } catch (e) {
                    const errorText = await response.text();
                    error = errorText || 'Conversion failed';
                }
                throw new Error(error);
            }

            // Get the image blob from the response
            const imageBlob = await response.blob();
            const imageUrl = URL.createObjectURL(imageBlob);

            // Display the converted image with slide reveal
            displayConvertedImage(imageUrl);

        } catch (error) {
            console.error('Error:', error);
            showError(error.message || 'An error occurred during conversion. Please try again.');
        }
    }

    /**
     * Display the original uploaded image
     * @param {string} imageUrl - URL of the original image
     */
    function displayOriginalImage(imageUrl) {
        const { result, preview } = initializeResultContainer();

        // Clear previous results and show the preview section
        preview.innerHTML = '';
        result.classList.remove('hidden');

        // Create container for images
        const container = document.createElement('div');
        container.className = 'image-container';

        // Create and append the original image
        const img = document.createElement('img');
        img.src = imageUrl;
        img.alt = 'Original Image';
        img.className = 'original-image';
        container.appendChild(img);

        // Add loading placeholder for the converted image
        const loadingPlaceholder = document.createElement('div');
        loadingPlaceholder.className = 'loading-placeholder';
        container.appendChild(loadingPlaceholder);

        // Add container to the preview
        preview.appendChild(container);

        // Scroll to result
        result.scrollIntoView({ behavior: 'smooth' });
    }

    /**
     * Display the converted image with slide reveal
     * @param {string} imageUrl - URL of the converted image
     */
    function displayConvertedImage(imageUrl) {
        const { result, preview } = initializeResultContainer();
        const container = preview.querySelector('.image-container');
        if (!container) {
            console.error('Image container not found');
            return;
        }

        // Remove loading placeholder
        const loadingPlaceholder = container.querySelector('.loading-placeholder');
        if (loadingPlaceholder) {
            loadingPlaceholder.remove();
        }

        // Create and append the converted image
        const convertedImg = document.createElement('img');
        convertedImg.src = imageUrl;
        convertedImg.alt = 'Converted ASCII Art';
        convertedImg.className = 'converted-image';
        container.appendChild(convertedImg);

        // Trigger the transition after a small delay
        setTimeout(() => {
            convertedImg.classList.add('visible');
            addActionButtons(imageUrl);
        }, 50);
    }

    /**
     * Add action buttons below the image
     * @param {string} imageUrl - URL of the converted image
     */
    function addActionButtons(imageUrl) {
        const { result, preview } = initializeResultContainer();
        if (!preview) return;

        // Create button container
        const buttonContainer = document.createElement('div');
        buttonContainer.className = 'action-buttons mt-6';

        // Create download button
        const downloadBtn = document.createElement('a');
        downloadBtn.href = imageUrl;
        downloadBtn.download = 'ascii-art.png';
        downloadBtn.className = 'px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded transition-colors';
        downloadBtn.textContent = 'Download Image';

        // Create back button
        const backBtn = document.createElement('button');
        backBtn.textContent = 'Convert Another';
        backBtn.className = 'ml-4 px-6 py-2 bg-gray-200 hover:bg-gray-300 rounded transition-colors';
        backBtn.onclick = resetConverter;

        // Add buttons to container
        buttonContainer.appendChild(downloadBtn);
        buttonContainer.appendChild(backBtn);

        // Add container to the page
        preview.appendChild(buttonContainer);

        // Fade in buttons
        setTimeout(() => buttonContainer.classList.add('visible'), 50);
    }

    /**
     * Reset the converter to its initial state
     */
    function resetConverter() {
        const uploadForm = document.getElementById('uploadForm');
        const result = document.getElementById('result');
        const fileInput = document.getElementById('imageInput');
        const fileInfo = document.getElementById('fileInfo');

        // Reset form
        if (fileInput) fileInput.value = '';
        if (fileInfo) fileInfo.textContent = '';

        // Hide result and show upload form
        if (result) {
            result.innerHTML = '';
            result.classList.add('hidden');
        }

        if (uploadForm && uploadForm.parentElement) {
            uploadForm.parentElement.classList.remove('hidden');
        }

        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    /**
     * Copy ASCII art to clipboard
     */
    function copyToClipboard() {
        const range = document.createRange();
        range.selectNode(asciiOutput);
        window.getSelection().removeAllRanges();
        window.getSelection().addRange(range);

        try {
            const successful = document.execCommand('copy');
            if (successful) {
                showFeedback('Copied to clipboard!');
            } else {
                throw new Error('Copy failed');
            }
        } catch (err) {
            console.error('Failed to copy text: ', err);
            showFeedback('Failed to copy to clipboard', true);
        }

        window.getSelection().removeAllRanges();
    }

    /**
     * Show feedback message to user
     * @param {string} message - The message to display
     * @param {boolean} isError - Whether the message is an error
     */
    function showFeedback(message, isError = false) {
        const feedbackDiv = document.createElement('div');
        feedbackDiv.className = `fixed top-4 right-4 px-4 py-2 rounded shadow-lg ${
            isError ? 'bg-red-500' : 'bg-green-500'
        } text-white`;
        feedbackDiv.textContent = message;
        document.body.appendChild(feedbackDiv);

        // Remove feedback after 3 seconds
        setTimeout(() => {
            feedbackDiv.remove();
        }, 3000);
    }

    /**
     * Show error message to user
     * @param {string} message - The error message to display
     */
    function showError(message) {
        showFeedback(message, true);
    }

// Add this at the end of the file, just before the closing }); of DOMContentLoaded

// Glitch effect for the title
function initGlitchEffect() {
    const titleElement = document.querySelector('.title-converter');
    if (!titleElement) return;

    const numFrames = 9; // Frames 00-08
    const baseFrameDuration = 70; // Base duration in milliseconds

    let isGlitching = false;
    let glitchTimeout;

    // Function to show a single frame, or remove the background image at the end of the sequence
    const showFrame = (frameNum) => {
        if(frameNum===false){
            attrValue = 'none';
        } else {
            const frameStr = frameNum.toString().padStart(2, '0');
            attrValue = `url(/static/img/converter-ascii-${frameStr}.png)`;
        }
        titleElement.style.backgroundImage = attrValue;
    };

    // Function to play the glitch sequence
    const playGlitchSequence = () => {
        if (isGlitching) return;
        isGlitching = true;

        // If this is the first glitch, switch to image mode
        titleElement.classList.add('title-converter-image-mode');
        titleElement.classList.remove('title-converter');

        let frameCount = 0;
        let currentFrame = 0;

        // Show the first frame immediately
        showFrame(currentFrame);

        // Set up the total number of frames
        const totalFrames = 3 + Math.floor(Math.random() * 17); // 3-20 frames

        const showNextFrame = () => {
            if (frameCount >= totalFrames - 1) {
                // End of the sequence
                isGlitching = false;

                titleElement.classList.add('title-converter');
                titleElement.classList.remove('title-converter-image-mode');
                showFrame(false);

                // Schedule the next glitch sequence
                const nextGlitchDelay = 2000 + Math.random() * 7000; // 2-9 seconds
                setTimeout(playGlitchSequence, nextGlitchDelay);
                return;
            }

            isGlitching = true
            frameCount++;
            currentFrame = Math.floor(Math.random() * numFrames);
            showFrame(currentFrame);

            // Calculate random delay for the next frame
            randBaseDelay = Math.floor(baseFrameDuration * (Math.random()+0.4));
            let randomDelay = randBaseDelay + Math.floor(Math.random() * baseFrameDuration * 2.5);

            // Schedule next frame
            setTimeout(showNextFrame, randomDelay);
        };
        if (isGlitching = true) {
            showNextFrame();
        }
    };

    // Schedule the next glitch
    const scheduleNextGlitch = () => {
        const delay = 3000 + Math.random() * 7000; // 8-16 seconds frequency
        clearTimeout(glitchTimeout);
        glitchTimeout = setTimeout(playGlitchSequence, delay);
    };

    // Initialize
    // Start with just text, schedule first glitch
    scheduleNextGlitch();
}

// Initialize the glitch effect when the page loads
initGlitchEffect();
});
