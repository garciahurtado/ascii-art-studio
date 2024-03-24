window.onload = function() {
    var canvas=document.getElementById("canvas");
    var buffer_el=document.getElementById("buffer");
    var start_time = performance.now();
    var start_frame_time = performance.now();
    var end_time = performance.now();

    var ctx=canvas.getContext('2d');
    ctx.imageSmoothingEnabled = false;

    // Hidden canvas
    var buffer=buffer_el.getContext('2d');
    buffer.imageSmoothingEnabled = false;

    var all_chars = [];
    var char_width = 8;
    var char_height = 8;

    async function frame_reader() {
        console.log("Received frame data")
        start_frame_time = performance.now();

        all_blocks = [];
        fg_colors = [];
        bg_colors = [];

        re = /dim=(\d+,\d+),,([^|]+)|/g;
        frames = this.responseText.matchAll(re);

        for(const frame of frames){
//            frame = frames[i]
            dims = frame[1];
            data = frame[2];

            [width, height] = dims.split(",");
            num_blocks = width * height

            var total_cols = width;
            var total_rows = height;

            all_blocks = decodeFrameData(data, num_blocks)

            // Draw the ASCII character blocks, and save the FG and BG colors
            for(index in all_blocks){
                block = all_blocks[index];
                col = Math.floor(index % total_cols);
                row = Math.floor(index / total_cols);

                drawBlock(ctx, col, row, block.char_index, all_chars, char_width, char_height);

                fg_colors.push(block.fg_color)
                bg_colors.push(block.bg_color)
            }

            console.log("Decoded frame")

            pixel_zoom = 2;
            [pixel_width, pixel_height] = [total_cols * char_width * pixel_zoom, total_rows * char_height * pixel_zoom]

            // Store the ASCII image in the hidden canvas, but inverted
            var ascii = ctx.getImageData(0, 0, pixel_width, pixel_height);
            var ascii_img = await createImageBitmap(ascii);

            buffer.drawImage(ascii_img, 0, 0);
            buffer.globalCompositeOperation='difference';
            buffer.fillStyle='white';
            buffer.fillRect(0, 0, pixel_width, pixel_width);

            // Draw the FG colors
            var fgImageData = ctx.createImageData(total_cols, total_rows);

            for(var i in fg_colors){
                color = fg_colors[i]

                fgImageData.data[i*4]   = color[0];   // Red
                fgImageData.data[i*4+1] = color[1];   // Green
                fgImageData.data[i*4+2] = color[2];   // Blue
                fgImageData.data[i*4+3] = 255;        // Alpha
            }
            ctx.scale(8,8)
            ctx.globalCompositeOperation = 'multiply'

            var fg_bitmap = await createImageBitmap(fgImageData);
            ctx.drawImage(fg_bitmap, 0, 0, total_cols, total_rows);

            // Draw the BG colors in the hidden canvas
            var bgImageData = buffer.createImageData(total_cols, total_rows);

            for(var i in bg_colors){
                color = bg_colors[i]

                bgImageData.data[i*4]   = color[0];   // Red
                bgImageData.data[i*4+1] = color[1];   // Green
                bgImageData.data[i*4+2] = color[2];   // Blue
                bgImageData.data[i*4+3] = 255;        // Alpha
            }
            buffer.scale(16,16)
            buffer.globalCompositeOperation = 'multiply'

            var bg_bitmap = await createImageBitmap(bgImageData);
            buffer.drawImage(bg_bitmap, 0, 0, total_cols, total_rows);

            // Finally, mix the two canvases together
            final_img = await getCanvasBitmap(buffer, pixel_width, pixel_height);
            ctx.globalCompositeOperation='lighten';
            ctx.drawImage(final_img, 0, 0, total_cols, total_rows);

            end_time = performance.now();
            elapsed = end_time - start_time;
            console.log("Total script time: " + elapsed + "ms");

            elapsed = end_time - start_frame_time;
            console.log("Frame render time: " + elapsed + "ms");
        }
    }

    charset = document.getElementById("charset");
    charset.crossOrigin = "Anonymous";
    charset.src = "img/amstrad-cpc.png"; // can also be a remote URL e.g. http://

    // Chop up the charset into characters, and create a bitmap for each
    charset.onload = function() {
        ctx.drawImage(charset,0,0);

        var height = charset.naturalHeight;
        var width = charset.naturalWidth;
        var total_rows = height / char_height;
        var total_cols = width / char_width;
        console.log("Reading a charset with " + width + " x " + height + " pixels")

        // Iterate through rows, then columns of character blocks in the charset
        index = 0;
        for(let y=0; y < width ; y += char_height){
           for(let x=0; x < height ; x += char_width){
               var img_data = ctx.getImageData(x, y, char_width, char_height);
               var char_img = createImageBitmap(img_data);
               char_img.index = index++;
               all_chars.push(char_img)
           }
        }

        // Add the blank character, not part of the charmap PNG
        var raw = new Uint8ClampedArray(char_width * char_height * 4); // 4 RGBA
        var blank_bitmap = createImageBitmap(new ImageData(raw, char_width, char_height));
        all_chars.push(blank_bitmap)

        Promise.allSettled(all_chars).then((results) => {
            console.log("Loaded charset");
            new_chars = [];
            for(index in results){
                new_chars.push(results[index].value)
            }
            all_chars = new_chars;

            ctx.scale(2,2);

            var req1 = new XMLHttpRequest();
            req1.addEventListener("load", frame_reader);
            req1.open("GET", "frame_data.txt");
            req1.send();
        });
    };
};

/* Functions ---------------------------*/

/**
 * Draw a single ASCII block, in black and white
 */
function drawBlock(ctx, col, row, char_index, all_chars, char_width, char_height) {
    char = all_chars[char_index];
    x = col * char_width;
    y = row * char_height;
    ctx.drawImage(char, x, y, char_width, char_height);
}

/**
 * Parses an arbitrarily long string of characters / colors from an ASCII image frame,
 * and returns a list of block objects.
 *
 * The format is:
 * <ascii_index>,[<fg color RGB array>],[<bg color RGB array>],...
 */
function decodeFrameData(data, num_blocks){

    // 256,[0,0,0],[0,0,0] represents one block (ascii, rgb, rgb)
    re = /(\d+),\[(\d+),(\d+),(\d+)\],\[(\d+),(\d+),(\d+)\]/g;
    matches = [...data.matchAll(re)];
    var blocks = [];

    for(index in matches){
        match = matches[index];

        block = [];
        block.char_index = parseInt(match[1]);
        block.fg_color = [parseInt(match[2]), parseInt(match[3]), parseInt(match[4])];
        block.bg_color = [parseInt(match[5]), parseInt(match[6]), parseInt(match[7])];

        blocks.push(block);
    }

    return blocks;
}

async function getCanvasBitmap(canvas, width, height){
    var image_data = canvas.getImageData(0, 0, width, height);
    var bitmap = await createImageBitmap(image_data);

    return bitmap;
}
