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
    const charCols = document.getElementById('charCols');
    const charRows = document.getElementById('charRows');

    // Toggle advanced settings
    const toggleAdvancedBtn = document.getElementById('toggleAdvanced');
    const advancedSettings = document.getElementById('advancedSettings');
    const brightnessSlider = document.getElementById('brightness');
    const brightnessValue = document.getElementById('brightnessValue');
    const contrastSlider = document.getElementById('contrast');
    const contrastValue = document.getElementById('contrastValue');

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

    charCols.addEventListener('click', () => {
        charRows.disabled = true;
        charCols.disabled = false;
    });

    charRows.addEventListener('click', () => {
        charCols.disabled = true;
        charRows.disabled = false;
    });

    charCols.addEventListener('input', () => {
        /* calculate the value of the other field */
        if (charCols.value && !isNaN(charCols.value)) {
            charRows.value = charCols.value * (charRows.value / charCols.value);
            console.log(charRows.value);
        }
    });
    charRows.addEventListener('input', () => {
        /* calculate the value of the other field */

        if (charRows.value && !isNaN(charRows.value)) {
            /* First figure out the aspect ratio of the image */
            aspectRatio = image.width / image.height;

            charCols.value = charRows.value * (charCols.value / charRows.value);
            console.log(charCols.value);
        }
    });

    if (toggleAdvancedBtn && advancedSettings) {
        toggleAdvancedBtn.addEventListener('click', () => {
            const isHidden = advancedSettings.classList.toggle('hidden');
            toggleAdvancedBtn.textContent = isHidden ? '▼ Advanced Settings' : '▲ Advanced Settings';
        });
    }

    // Update slider value displays
    if (brightnessSlider && brightnessValue) {
        brightnessSlider.addEventListener('input', () => {
            brightnessValue.textContent = brightnessSlider.value;
        });
    }

    if (contrastSlider && contrastValue) {
        contrastSlider.addEventListener('input', () => {
            contrastValue.textContent = contrastSlider.value;
        });
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
        // Add advanced settings to form data if they exist
        if (brightnessSlider) formData.append('brightness', brightnessSlider.value);
        if (contrastSlider) formData.append('contrast', contrastSlider.value);

        // Show loading state and prepare result container
        uploadForm.parentElement.classList.add('hidden');

        if (result) {
            result.innerHTML = '';
            result.classList.remove('hidden');

            addClassAuto = '';
            removeClassAuto = '';

            if (charCols >= charRows) {
                /* Landscape */
                addClassAuto = 'auto-landscape';
                removeClassAuto = 'auto-portrait';
            } else {
                /* Portrait */
                addClassAuto = 'auto-portrait';
                removeClassAuto = 'auto-landscape';
            }
            result.classList.add(addClassAuto);
            result.classList.remove(removeClassAuto);
            result.classList.add('hidden');
        }

        // Display original image immediately
        const originalImageUrl = URL.createObjectURL(file);
        let response;  // Declare response here to make it accessible in catch
        displayOriginalImage(originalImageUrl);

        try {
            // Send the POST request
            response = await fetch('/convert', {
                method: 'POST',
                body: formData
            });

            // First check if the response is ok (status in the range 200-299)
            if (!response.ok) {
                // Try to parse the error response as JSON
                let errorData;
                try {
                    errorData = await response.json();
                } catch (e) {
                    // If we can't parse JSON, use the status text
                    throw new Error(`Server error: ${response.status} ${response.statusText}`);
                }
                throw errorData;
            }

            // If we get here, the request was successful
            const imageBlob = await response.blob();
            const imageUrl = URL.createObjectURL(imageBlob);
            displayConvertedImage(imageUrl);

        } catch (error) {
            console.error('Conversion error:', error);

            let errorMessage = 'An error occurred during conversion';

            if (error instanceof Error) {
                // Handle standard Error objects
                errorMessage = error.message;
            } else if (error && typeof error === 'object') {
                // Handle error objects with validation errors
                if (error.errors) {
                    errorMessage = "There were validation errors:\n";
                    for (const key in error.errors) {
                        errorMessage += `${key}: ${Array.isArray(error.errors[key]) ?
                            error.errors[key].join(', ') : error.errors[key]}\n`;
                    }
                } else if (error.detail) {
                    errorMessage = error.detail;
                } else if (error.message) {
                    errorMessage = error.message;
                }
            } else if (response) {
                // Try to get error from response if available
                try {
                    const errorData = await response.json();
                    if (errorData.errors) {
                        errorMessage = "There were validation errors:\n";
                        for (const key in errorData.errors) {
                            errorMessage += `${key}: ${Array.isArray(errorData.errors[key]) ?
                                errorData.errors[key].join(', ') : errorData.errors[key]}\n`;
                        }
                    } else if (errorData.detail) {
                        errorMessage = errorData.detail;
                    }
                } catch (e) {
                    // If we can't parse JSON, use status text
                    errorMessage = `Server error: ${response.status} ${response.statusText}`;
                }
            }

            // Show the error to the user
            showError(errorMessage, 'Conversion Error');
        }
    }

    /**
     * When we click on either of the columns or rows fields, we need to disable the other one
     */
    function attachFieldListeners() {

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
     * @param {string|Error} error - The error message or Error object
     * @param {string} [title='Error'] - The title of the error message
     */
    function showError(error, title = 'Error') {
        const errorDiv = document.getElementById('error-message');
        if (!errorDiv) {
            console.error('Error message container not found');
            return;
        }

        // Get the error message from Error object if needed
        const errorMessage = error instanceof Error ? error.message : String(error);

        // Log to console with styling
        console.error(
            `%c${title}:%c ${errorMessage}`,
            'color: white; background: #dc3545; padding: 2px 8px; border-radius: 4px; font-weight: bold;',
            'color: #dc3545;',
        );

        // Get the template and clone it
        const template = document.getElementById('toast-template');
        if (!template) {
            console.error('Toast template not found');
            return;
        }

        const toast = template.content.cloneNode(true);
        const toastElement = toast.querySelector('.toast');
        const toastId = 'toast-' + Date.now();

        // Set the toast ID
        toastElement.id = toastId;

        // Set the title and message
        const titleElement = toastElement.querySelector('.toast-title');
        const bodyElement = toastElement.querySelector('.toast-body');

        if (titleElement) titleElement.textContent = title;
        if (bodyElement) bodyElement.innerHTML = errorMessage.replace(/\n/g, '<br>');

        // Add close button handler
        const closeButton = toastElement.querySelector('.toast-close');
        if (closeButton) {
            closeButton.addEventListener('click', () => {
                hideToast(toastElement);
            });
        }

        // Add to DOM
        errorDiv.appendChild(toast);

        // Auto-remove the toast after 8 seconds
        const timeoutId = setTimeout(() => {
            hideToast(toastElement);
        }, 8000);

        // Store timeout ID on element for cleanup
        toastElement._timeoutId = timeoutId;
    }

    function hideToast(toastElement) {
        if (!toastElement) return;

        // Clear the auto-hide timeout
        if (toastElement._timeoutId) {
            clearTimeout(toastElement._timeoutId);
        }

        // Add fade-out animation
        toastElement.style.animation = 'toast-fade-out 0.5s ease-in forwards';

        // Remove from DOM after animation completes
        setTimeout(() => {
            if (toastElement && toastElement.parentNode) {
                toastElement.parentNode.removeChild(toastElement);
            }
        }, 500);
    }
});
