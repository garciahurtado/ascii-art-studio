class AsciiMoviePlayer {
    constructor(ctx, video_file, char_width, char_height) {
        var pixel_zoom = 2;
        var scale = pixel_zoom * 2
        this.scale = scale;

        var doc = ctx.canvas.ownerDocument;
        this.ctx = ctx;
        this.video_file = video_file;

        var num_cols = 80;
        var num_rows = 45;

        this.ctx.canvas.width = (num_cols * 8) * pixel_zoom;
        this.ctx.canvas.height = (num_rows * 8) * pixel_zoom;
        this.ctx.scale(pixel_zoom, pixel_zoom);
        ctx.imageSmoothingEnabled = false;

        this.fg_layer;
        this.bg_layer;
        this.palette;

        this.frame_chars = [];
        this.frame_counter = doc.getElementById('frame_counter');
        this.charset;
        this.char_width;
        this.char_height;
        this.full_frame; // bitmap of the last full frame, to composite with partial frames
        this.total_cols;
        this.total_rows;
        this.video_stream;
        this.is_paused = false;
        this.fps = 24;
        this.time_last_frame;
        this.all_frames_time = 0;
        this.total_frames;
        this.runner; // FPS runner
        this.show_grid = false;
        this.grid_color = '#50905033';

        this.playhead = doc.getElementById('playhead');
        this.prog_bar = doc.getElementById('prog_bar');
        this.playbar = this.playhead.parentElement;

        var palette_canvas = doc.getElementById('palette_canvas');
        this.palette_canvas = palette_canvas.getContext('2d');
        this.palette_canvas.scale(pixel_zoom, pixel_zoom);
        this.palette_canvas.imageSmoothingEnabled = false;
        this.show_palette = false;

        this.play_button;
        this.pause_button;

        // Create a second hidden canvas, to use as a buffer for loading the charset
        var fg_layer_el = doc.createElement('canvas');
        fg_layer_el.id = 'fg_layer';
        fg_layer_el.style.display="none";
        this.fg_layer = fg_layer_el.getContext('2d', { willReadFrequently: true });
        this.fg_layer.imageSmoothingEnabled = false;
        doc.body.appendChild(fg_layer_el);

        // And a third one for mixing ASCII character masks and colors
        var bg_layer_el = doc.createElement('canvas', {'id': 'bg_layer'});
        bg_layer_el.id = 'bg_layer';
        doc.body.appendChild(bg_layer_el);
        bg_layer_el.style.display="none";
        this.bg_layer = bg_layer_el.getContext('2d', { willReadFrequently: true });

        this.fg_layer.canvas.width = (char_width * num_cols);
        this.fg_layer.canvas.height = (char_height * num_rows);

        this.bg_layer.canvas.width = (char_width * num_cols);
        this.bg_layer.canvas.height = (char_height * num_rows);

        this.bg_layer.imageSmoothingEnabled = false;

    }

    /**
     * Given a list of decoded blocks, including ASCII character codes and colors, draw them on the visible canvas
     */
    async drawFrame(frame){
        var start_frame_time = performance.now();

        this.total_cols = frame.cols;
        this.total_rows = frame.rows;
        var char_width = frame.char_width;
        var char_height = frame.char_height;

        var pixel_width;
        var pixel_height;
        [pixel_width, pixel_height] = [frame.cols * char_width, frame.rows * char_height]

        this.ctx.globalCompositeOperation='source-over';

        this.fg_layer.globalCompositeOperation='source-over';
        this.fg_layer.clearRect(0, 0, pixel_width, pixel_height);

        this.bg_layer.globalCompositeOperation='source-over';
        this.bg_layer.clearRect(0, 0, pixel_width, pixel_height);

        var start_time = performance.now();

        if(frame.length == 0){
            return false;
        }

        // Create a blank pixel array to draw the characters on top of
        var ascii_img = new ImageData(new Uint8ClampedArray(pixel_width * pixel_height * 4), pixel_width, pixel_height);

        // Draw the ASCII character blocks
        for(var index in frame.blocks){
            var block = frame.blocks[index];
            this.drawBlock(ascii_img, block.x, block.y, block.char_index, char_width, char_height);
        }
        var bitmap = await createImageBitmap(ascii_img);
        this.fg_layer.drawImage(bitmap, 0, 0);

        // Store the ASCII image in the hidden canvas, but inverted
        var ascii = this.fg_layer.getImageData(0, 0, pixel_width, pixel_height);
        var ascii_img = await createImageBitmap(ascii);

        // Clear mask buffer
        this.bg_layer.filter = 'invert(1)';
        this.bg_layer.drawImage(ascii_img, 0, 0);
        this.bg_layer.filter = 'invert(0)';

        var scale = char_width; // 8x8 blocks
        await Promise.all([
            this.drawColorBlocks(frame.fg_colors, this.fg_layer, scale),
            this.drawColorBlocks(frame.bg_colors, this.bg_layer, scale)])

        // Finally, mix the two canvases together (FG colors and BG colors)
        var bg_img = await this.getCanvasBitmap(this.bg_layer, pixel_width, pixel_height);
        this.fg_layer.globalCompositeOperation='lighten';
        this.fg_layer.drawImage(bg_img, 0, 0,  pixel_width, pixel_height);

        var new_img = await this.getCanvasBitmap(this.fg_layer, pixel_width, pixel_height);

        // Finally, draw the final image on the visible canvas and take a snapshot
        this.ctx.drawImage(new_img, 0, 0,  pixel_width, pixel_height);

        if(this.show_grid){
           this.drawGrid();
        }

        if(this.show_palette){
            this.drawPalette(this.palette.colors);
        }

        var end_time = performance.now();
        var elapsed = end_time - start_frame_time;
        this.all_frames_time += elapsed;
        //console.log("Frame render time: " + elapsed + "ms");
    }


    /**
     * Draw a single ASCII block within the given image data, in black and white
     */
    drawBlock(img_data, col, row, char_index, char_width, char_height) {
        var char = this.charset[char_index];

        for (var y = 0; y < char_height; y++) {
            for (var x = 0; x < char_width; x++) {
                var char_pixel = ((y * char_width) + x) * 4; // 4 bytes per pixel
                var pos_x = (col * char_width) + x;
                var pos_y = (row * char_height) + y;
                var pos = ((pos_y * img_data.width) + pos_x) * 4;

                img_data.data[pos +0] = char.data[char_pixel +0];
                img_data.data[pos +1] = char.data[char_pixel +1];
                img_data.data[pos +2] = char.data[char_pixel +2];
                img_data.data[pos +3] = 255;
            }
        }
    }

    /**
     * Draw a series of 8x8 color blocks onto a specified canvas, and at the given scale
     */
    async drawColorBlocks(blocks, my_canvas, scale){
        my_canvas.save();
        var imageData = await my_canvas.createImageData(this.total_cols, this.total_rows);

        var color;
        var block_start;
        var pos = 0;
        var i = 0;

        for(i in blocks){
            color = blocks[i];
            block_start = i * 4;

            imageData.data[block_start] = color[2];   // Red
            imageData.data[block_start + 1] = color[1];   // Green
            imageData.data[block_start + 2] = color[0];   // Blue
            imageData.data[block_start + 3] = 256;        // Alpha
        }
        var bitmap = await createImageBitmap(imageData);

        my_canvas.scale(scale, scale);
        my_canvas.imageSmoothingEnabled = false;
        my_canvas.globalCompositeOperation = 'multiply';

        my_canvas.drawImage(bitmap, 0, 0, this.total_cols, this.total_rows);
        my_canvas.restore(); // restore scale
    }


    /**
     * Returns a bitmap screenshot of the passed canvas
     */
    async getCanvasBitmap(canvas, width, height){
        var image_data = canvas.getImageData(0, 0, width, height);
        var bitmap = await createImageBitmap(image_data);

        return bitmap;
    }

    /**
     * From an image, extract bitmap slices that are stored in an array as a
     * character set, later used to render images onto the canvas.
     */
    async loadCharset(charset_image, char_width, char_height, callback){
        this.char_width, this.char_height = char_width, char_height;
        this.fg_layer.globalCompositeOperation='source-over';
        this.fg_layer.drawImage(charset_image,0,0);

        this.char_width = char_width;
        this.char_height = char_height;

        var height = charset_image.naturalHeight;
        var width = charset_image.naturalWidth;
        this.total_rows = height / char_height;
        this.total_cols = width / char_width;
        console.log("Reading a charset with " + width + " x " + height + " pixels / " +
            this.total_rows + " rows, " + this.total_cols + " cols")
        console.log("Reading a charset with " + width + " x " + height + " pixels")


        // Iterate through rows, then columns of character blocks in the charset
        var charset = [];
        var index = 0;
        var img_data;
        var char_img;

        for(let row=0; row < this.total_rows ; row++){
            var y = row * char_height;
            for(let col=0; col < this.total_cols ; col++){
               var x = col * char_width;
               img_data = this.fg_layer.getImageData(x, y, char_width, char_height);
               charset.push(img_data);
           }
        }

        Promise.allSettled(charset).then((results) => {
            console.log("Loaded charset with " + index + " characters.");
            var char_imgs = [];
            results.forEach(result => {
                var mychar = result.value;
                char_imgs.push(mychar);
            });

            this.charset = char_imgs;
            this.loadVideoStream(this.video_file);
        });
    }

    drawGrid(){
        var height = this.ctx.canvas.height;
        var width = this.ctx.canvas.width;

        this.ctx.beginPath();
        this.ctx.lineWidth = 1;
        this.ctx.strokeStyle = this.grid_color;

        // Vertical lines
        for(var col=0; col < this.total_cols; col++){
            this.ctx.moveTo(col * this.char_width, 0);
            this.ctx.lineTo(col * this.char_width, height);
        }
        // Horizontal lines
        for(var row=0; row < this.total_rows; row++){
            this.ctx.moveTo(0, row * this.char_height);
            this.ctx.lineTo(width, row * this.char_height);
        }
        this.ctx.stroke();
    }

    drawPalette(colors){
        var swatch_size = 10;
        var cols = 32;
        var rows = 8;

        var palette = this.palette_canvas;
        palette.canvas.width = swatch_size * cols;
        palette.canvas.height = swatch_size * rows;
        palette.globalCompositeOperation='source-over';
        palette.clearRect(0, 0, palette.canvas.width, palette.canvas.height);

        for(var i=0; i < colors.length; i++){
            var color = colors[i];
            var x = (i % cols) * swatch_size;
            var y = Math.floor(i / cols) * swatch_size;
            palette.fillStyle = 'rgb(' + parseInt(color[2]) + ',' + parseInt(color[1]) + ',' + parseInt(color[0]) + ')';
            palette.fillRect(x, y, swatch_size - 2, swatch_size - 2); // leave a gap
            palette.fill();
        }
    }

    async loadVideoStream(url){
        var req = new XMLHttpRequest();
        var player = this;

        req.addEventListener("loadend", async function (e) {
            try {
                if (this.status >= 200 && this.status < 300 && this.response) {
                    player.video_stream = this.response;
                    await player.parseVideoStream();
                } else {
                    const error = new Error(`Failed to load video: ${this.statusText || 'Unknown error'}`);
                    error.status = this.status;
                    throw error;
                }
            } catch (error) {
                console.error('Error in loadVideoStream:', error);
                // Rethrow to be caught by the global error handler
                setTimeout(() => { throw error; });
            }
        });

        req.onerror = function() {
            const error = new Error(`Network error while loading video: ${url}`);
            console.error(error);
            setTimeout(() => { throw error; });
        };

        req.onabort = function() {
            const error = new Error('Video loading was aborted');
            console.error(error);
            setTimeout(() => { throw error; });
        };

        try {
            req.open("GET", url);
            req.responseType = "arraybuffer";
            req.send();
        } catch (error) {
            console.error('Error sending video request:', error);
            setTimeout(() => { throw error; });
        }
    }

    async parseVideoStream(){
        try {
            var start_time = performance.now();
            this.all_frames_time = 0;

            if (!this.video_stream || !this.video_stream.byteLength) {
                throw new Error('No video data available to parse');
            }

            var all_bytes = new Uint8Array(this.video_stream);
            var header = this.decoder.decodeStreamHeader(all_bytes);

            if (!header || !header.num_frames) {
                throw new Error('Invalid video header or no frames found');
            }

            this.total_frames = header.num_frames;
            var all_frames = this.decoder.readAllFrames(all_bytes, this.total_frames, header.char_width, header.char_height);

            if (!all_frames || !all_frames.length) {
                throw new Error('No frames could be decoded from the video');
            }

            this.runner = new FpsRunner(this.fps);
            this.runner.frames = all_frames;
            var self = this;

            this.runner.callback = async (frame) => {
                try {
                    await this.showFrame(frame);
                } catch (error) {
                    console.error('Error in frame callback:', error);
                    // Rethrow to be caught by the global error handler
                    setTimeout(() => { throw error; });
                }
            };

            requestAnimationFrame(function(){
                try {
                    self.runner.play();
                } catch (error) {
                    console.error('Error starting playback:', error);
                    setTimeout(() => { throw error; });
                }
            });

            this.runner.on_finish = function() {
                try {
                    self.stop();
                    var end_time = performance.now();
                    var elapsed = end_time - start_time;
                    console.log("*** All frames time: " + self.all_frames_time + "ms");

                    var avg = self.all_frames_time / self.total_frames;
                    console.log("*** " + self.total_frames + " frames @ " + avg + "ms per frame");
                } catch (error) {
                    console.error('Error in on_finish:', error);
                    setTimeout(() => { throw error; });
                }
            };
        } catch (error) {
            console.error('Error parsing video stream:', error);
            // Rethrow to be caught by the global error handler
            setTimeout(() => { throw error; });
        }
    }

    /**
     * Renders a single frame, passed in the CPEG format
     */
    async showFrame(frame){
        var start_time = performance.now();
        this.palette = frame.palette;

        await this.drawFrame(frame);
        this.frame_counter.value = this.runner.frame_index + 1; // Frame numbers start at 1 in the UI
        this.updatePlayhead(this.runner.frame_index);
        this.all_frames_time += performance.now() - start_time;
    }

    play(){
        this.is_paused = false;
        this.runner.play();
        this.pause_button.style.visibility = 'visible';
        this.play_button.style.visibility = 'hidden';
    }

    pause(){
        if(this.is_paused){
            this.play();

        } else{
            this.stop();
        }
    }

    stop(){
        this.runner.stop();
        this.is_paused = true;
        this.pause_button.style.visibility = 'hidden';
        this.play_button.style.visibility = 'visible';
    }

    reset(){
        this.runner.frame_index = 0;
    }

    /**
     * When the user clicks on the scrub bar, we have to figure out which frame that
     * screen position is equivalent to
     */
    seekToPosition(x){
        var total_width = this.playbar.clientWidth - this.playhead.style.width;

        // substract the width of the playhead dot, so it doesn't go off screen
        total_width -= 8;

        var percent = x / total_width;
        var frame_num = this.total_frames * percent;

        this.goToFrame(Math.round(frame_num), this.scale);
    }

    goToFrame(frame_num){
        if(frame_num < 0){
            frame_num = 0;
        } else if(frame_num > this.total_frames - 1){
            frame_num = this.total_frames - 1;
        }
        this.runner.frame_index = frame_num;
        this.updatePlayhead(frame_num);

        if(!this.runner.running){
            this.showFrame(this.runner.getCurrentFrame());
        }
    }

    goToNextFrame(){
        var frame_num = this.runner.frame_index + 1;
        if(frame_num > this.total_frames - 1){
            frame_num = this.total_frames - 1;
        }

        this.goToFrame(frame_num);
    }

    goToPrevFrame(){
        var frame_num = this.runner.frame_index - 1;

        if(frame_num < 0){
            frame_num = 0;
        }

        this.goToFrame(frame_num);
    }

    updatePlayhead(frame_num){
        if(frame_num > this.total_frames){
            return;
        }
        var percent = frame_num / (this.total_frames - 1);
        var total_width = this.playbar.clientWidth - this.playhead.style.width;

        // substract the width of the playhead dot, so it doesn't go off screen
        total_width -= 8;
        var current_width = (percent * total_width);
        this.playhead.style['margin-left'] = current_width + 'px';
        this.prog_bar.style.width = current_width + 'px';
    }
}