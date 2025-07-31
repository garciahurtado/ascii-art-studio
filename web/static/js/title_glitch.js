document.addEventListener('DOMContentLoaded', () => {
    // Glitch effect for the title
    async function initGlitchEffect() {
        const titleElement = document.querySelector('.title-converter');
        if (!titleElement) return;

        const glitchImage = document.querySelector('img.glitchy');
        if (!glitchImage) return;

        const numFrames = 9; // Frames 00-08
        const baseFrameDuration = 180; // Base duration in milliseconds

        // Preload all frames
        const frames = await Promise.all(
            Array.from({ length: numFrames }, (_, i) => {
                const frameNum = i.toString().padStart(2, '0');
                const img = new Image();
                img.src = `/static/img/converter-ascii-${frameNum}.png`;
                return new Promise((resolve) => {
                    img.onload = () => resolve(img);
                    img.onerror = () => resolve(null);
                });
            })
        );

        let isGlitching = false;
        let glitchTimeout;

        // Function to show a single frame
        const showFrame = (frameNum) => {
            if (frameNum >= 0 && frameNum < frames.length && frames[frameNum]) {
                glitchImage.src = frames[frameNum].src;
                return true;
            }
            return false;
        };

        // Function to play the glitch sequence
        const playGlitchSequence = () => {
            if (isGlitching) return;
            isGlitching = true;

            // If this is the first glitch, switch to image mode
            titleElement.classList.add('title-converter-image-mode');

            let frameCount = 0;
            let currentFrame = 0;

            // Show the first frame immediately
            showFrame(currentFrame);

            // Set up the total number of frames
            const totalFrames = 5 + Math.floor(Math.random() * 15); // x-y frames

            const showNextFrame = () => {
                if (frameCount >= totalFrames - 1) {
                    // End of the sequence
                    isGlitching = false;

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
                randBaseDelay = Math.floor(baseFrameDuration + ((baseFrameDuration/2) * (Math.random()-0.5)));

                // Schedule next frame
                setTimeout(showNextFrame, randBaseDelay);
            };
            if (isGlitching = true) {
                showNextFrame();
            }
        };

        // Schedule the next glitch
        const scheduleNextGlitch = () => {
            const delay = 4000 + Math.random() * 6000; // x-(x+y) seconds frequency
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