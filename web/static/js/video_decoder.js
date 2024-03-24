const FLAG_FULL_FRAME = 1;
const FLAG_DIFF_FRAME = 2;
const FLAG_PALETTE = 3;
const FLAG_HEADER = 4;

class VideoDecoder {
    constructor(){
        this.current_decoded_frame = 0;
        this.progress_el;
        this.header;
    }

    /**
     * Decode the stream header
     */
    decodeStreamHeader(bytes){
        var magic_number;   // 4 bytes
        var fps;            // 1 byte
        var num_frames;     // 2 bytes
        var duration;       // 2 bytes
        var width;          // 2 bytes
        var height;         // 2 bytes
        var char_width;     // 1 byte
        var char_height;    // 1 byte
        var charset_len;    // 1 byte
        var charset_name;   // x bytes
        var header_size = 17;

        magic_number = String.fromCharCode(...bytes.slice(0,4));

        if(magic_number !='CPEG'){
            throw("Invalid stream: CPEG Magic number not found at start of file.");
        }

        fps =           bytes[4];
        num_frames =    this.byteArrayToLong(bytes.slice(5, 8));
        duration =      this.byteArrayToLong(bytes.slice(8, 10));
        width =         this.byteArrayToLong(bytes.slice(10, 12));
        height =        this.byteArrayToLong(bytes.slice(12, 14));
        char_width =    bytes[14];
        char_height =   bytes[15];
        charset_len =   bytes[16];
        charset_name =  String.fromCharCode(...bytes.slice(17, 17+charset_len));
        header_size += charset_len;

        this.header = {
            fps: fps,
            num_frames: num_frames,
            duration: duration,
            width: width,
            height: height,
            char_width: char_width,
            char_height: char_height,
            charset_name: charset_name,
            header_size: header_size
        };

        return this.header;
    }


    readAllFrames(bytes, limit, char_width, char_height){
        var all_frames = this.splitFrames(bytes, this.header.header_size);
        var decoded_frames = [];

        var frame;
        for(var i in all_frames){
            i = parseInt(i);
            if(i + 1 > limit){
                break;
            }
            frame = all_frames[i];
            frame.char_width = char_width;
            frame.char_height = char_height;
            decoded_frames.push(frame);

            self = this;
            window.requestAnimationFrame(this.updateProgress.bind(this));
        }

        return decoded_frames;
    }

    updateProgress(){
        var total_frames = 217; // made up, should be read from video header in the future
        this.current_decoded_frame++;
        var pct = (this.current_decoded_frame / total_frames) * 100
        this.progress_el.style.width = parseInt(pct) + '%';
        // console.log("Percent: " + pct);
    }

    /**
     * Take a series of bytes and split it up into frames, and return them decoded
     */
    splitFrames(bytes, start=0){
        var all_frames = [];

        var palette;
        var header_size = 5;
        var data_size = 0;
        var frame_size = 0;
        var i = start;
        var header;

        while(header = this.decodeHeader(bytes, i)){
            frame_size = header.data_size;

            var frame_data = bytes.slice(i + header_size, i + header_size + frame_size);
            var new_frame = this.decodeFrame(frame_data, header.num_blocks, header.flags, header.width, header.height, palette);
            new_frame.header = header;

            if(new_frame.header.flags == FLAG_PALETTE){
                palette = new_frame;
            } else {
                all_frames.push(new_frame);
            }
            i = i + header_size + frame_size;
        }

        return all_frames;
    }

    /**
     * Decode a frame header (in bytes) and return an object with properties
     */
    decodeHeader(bytes, offset = 0){
        var flags;
        var width;
        var height;
        var num_blocks;
        var data_size;
        var block_size;

        if((offset + 5) > bytes.byteLength){
            return false;
        }

        [flags, width, height] = [bytes[offset+0], bytes[offset+1], bytes[offset+2]];
        var num_blocks = this.byteArrayToLong(bytes.slice(offset+3, offset+5));

        if(flags == FLAG_HEADER){

        } else if(flags == FLAG_FULL_FRAME){
            console.log("Frame type FLAG_FULL_FRAME: " + width + "x" + height);
            block_size = 4;
            num_blocks = width * height;
        } else if(flags == FLAG_DIFF_FRAME){
            console.log("Frame type FLAG_DIFF_FRAME");
            block_size = 6;
        } else if(flags == FLAG_PALETTE) {
            console.log("Frame type PALETTE (" + num_blocks + " colors)");
            block_size = 3;
        } else {
            throw 'INVALID FRAME TYPE'
        }
        data_size = num_blocks * block_size;

        return {
            flags: flags,
            width: width,
            height: height,
            num_blocks: num_blocks,
            data_size: data_size,
            block_size: block_size
        };
    }

    /**
     * Take a series of bytes representing a frame, and convert it to
     * a series of blocks representing characters and colors, and finally
     * wrap it up into a Frame object.
     */
    decodeFrame(bytes, num_blocks, flags, cols, rows, palette) {
        var data = bytes.slice(0, bytes.length);
        var block_size;

        if(flags == FLAG_FULL_FRAME){
            var frame = new VideoFrame(cols, rows, true);
            block_size = 4;
        } else if(flags == FLAG_DIFF_FRAME) {
            var frame = new VideoFrame(cols, rows, false);
            block_size = 6;
        } else if(flags == FLAG_PALETTE){
            var frame = new PaletteFrame();
            block_size = 3;
        } else {
            throw 'UNKNOWN FRAME TYPE';
        }
        var blocks = this.decodeFrameBlocks(data, num_blocks, block_size, flags);
        frame.palette = palette;
        frame.setBlocks(blocks);

        return frame;
    }

    /**
     * Parses an arbitrarily long string of characters / colors from an ASCII image frame,
     * and returns a list of block objects.
     *
     * The format is:
     * Absolute blocks (8 bytes):
     * <ascii_index>,[<fg color RGB array>],[<bg color RGB array>],...
     *
     * For relative blocks (10 bytes):
     * <ascii_index>,[<fg color RGB array>],[<bg color RGB array>],posX,posY...
     */
    decodeFrameBlocks(data, num_blocks, block_size, flags){
        var blocks = [];

        for(var index=0; index < (num_blocks * block_size); index += block_size){
            var block = [];

            if(flags == FLAG_PALETTE){
                block.red = data[index]
                block.green = data[index + 1]
                block.blue = data[index + 2]
            } else {
                block.char_index = this.byteArrayToLong([data[index], data[index+1]]);
                block.fg_color = parseInt(data[index+2]);
                block.bg_color = parseInt(data[index+3]);
            }

            if(flags == FLAG_DIFF_FRAME){
                block.x = data[index+4];
                block.y = data[index+5];
                block.bg_color.x = block.x;
                block.bg_color.y = block.y;
                block.fg_color.x = block.x;
                block.fg_color.y = block.y;
            }

            blocks.push(block);
        }

        return blocks;
    }

    byteArrayToLong(/*byte[]*/byteArray) {
        /**
         * Assumes little-endianness
         */
        var value = 0;
        for ( var i = byteArray.length - 1; i >= 0; i--) {
            value = (value * 256) + byteArray[i];
        }

        return value;
    };
}



