/**
 * UI Utilities for the ASCII Movie Player
 *
 * Provides common UI functionality like error handling and user feedback
 * that can be shared across different parts of the application.
 */

/**
 * Show a feedback message to the user
 * @param {string} message - The message to display
 * @param {boolean} isError - Whether this is an error message (default: false)
 * @param {number} duration - How long to show the message in ms (default: 3000)
 */
function showFeedback(message, isError = false, duration = 3000) {
    // Create the feedback element if it doesn't exist
    let feedbackContainer = document.getElementById('feedback-container');
    if (!feedbackContainer) {
        feedbackContainer = document.createElement('div');
        feedbackContainer.id = 'feedback-container';
        feedbackContainer.style.position = 'fixed';
        feedbackContainer.style.top = '20px';
        feedbackContainer.style.right = '20px';
        feedbackContainer.style.zIndex = '1000';
        document.body.appendChild(feedbackContainer);
    }

    // Create the message element
    const feedbackDiv = document.createElement('div');
    feedbackDiv.className = `feedback-message ${isError ? 'error' : 'success'}`;

    // Style the message
    feedbackDiv.style.padding = '12px 20px';
    feedbackDiv.style.marginBottom = '10px';
    feedbackDiv.style.borderRadius = '4px';
    feedbackDiv.style.color = 'white';
    feedbackDiv.style.boxShadow = '0 4px 6px rgba(0, 0, 0, 0.1)';
    feedbackDiv.style.opacity = '0';
    feedbackDiv.style.transition = 'opacity 0.3s ease-in-out';

    // Set background color based on message type
    if (isError) {
        feedbackDiv.style.backgroundColor = '#ef4444'; // red-500
    } else {
        feedbackDiv.style.backgroundColor = '#10b981'; // emerald-500
    }

    feedbackDiv.textContent = message;
    feedbackContainer.appendChild(feedbackDiv);

    // Trigger reflow to enable the fade-in animation
    void feedbackDiv.offsetWidth;
    feedbackDiv.style.opacity = '1';

    // Remove the message after the specified duration
    setTimeout(() => {
        feedbackDiv.style.opacity = '0';
        setTimeout(() => {
            feedbackDiv.remove();
            // Remove container if it's empty
            if (feedbackContainer.children.length === 0) {
                feedbackContainer.remove();
            }
        }, 300); // Match the CSS transition duration
    }, duration);
}

/**
 * Show an error message to the user
 * @param {string} message - The error message to display
 * @param {number} duration - How long to show the message in ms (default: 5000)
 */
function showError(message, duration = 5000) {
    console.error(message);
    showFeedback(message, true, duration);
}

/**
 * Show a success message to the user
 * @param {string} message - The success message to display
 * @param {number} duration - How long to show the message in ms (default: 3000)
 */
function showSuccess(message, duration = 3000) {
    console.log(message);
    showFeedback(message, false, duration);
}

/**
 * Handle a promise and show appropriate feedback
 * @param {Promise} promise - The promise to handle
 * @param {string} successMessage - Message to show on success
 * @param {string} errorMessage - Message to show on error (default: 'An error occurred')
 * @returns {Promise} The original promise for chaining
 */
function handlePromise(promise, successMessage, errorMessage = 'An error occurred') {
    return promise
        .then(result => {
            if (successMessage) {
                showSuccess(successMessage);
            }
            return result;
        })
        .catch(error => {
            const message = error instanceof Error ? error.message : errorMessage;
            showError(message);
            throw error; // Re-throw to allow further error handling
        });
}

// Export the functions for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    // For Node.js/CommonJS
    module.exports = {
        showFeedback,
        showError,
        showSuccess,
        handlePromise
    };
}
