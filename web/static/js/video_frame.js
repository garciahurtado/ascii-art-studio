class VideoFrame {
    constructor(cols, rows, is_full = true) {
        this.is_full = is_full;
        this.cols = cols;
        this.rows = rows;
        this.pixel_width;
        this.pixel_height;
        this.char_width;
        this.char_height;
        this.blocks;
        this.fg_colors;
        this.bg_colors;
        this.next_frame;
        this.prev_frame;
        this.palette;
    }

    /**
     * Reset the blocks, FG colors and BG colors arrays. Iterate through
     * the list of passed blocks, and collect the FG and BG colors for each.
     */
    setBlocks(blocks){
        this.blocks = [];
        this.fg_colors = [];
        this.bg_colors = [];

        for(var i in blocks){
            var block = blocks[i];

            if(this.is_full){ // full frame
                block.x = i % this.cols;
                block.y = Math.floor(i / this.cols);
            }

            //# Retrieve FG color from palette
            block.fg_color = this.palette.colors[block.fg_color]
            block.fg_color.x = parseInt(block.x);
            block.fg_color.y = parseInt(block.y);

            //# Retrieve BG color from palette by index
            block.bg_color = this.palette.colors[block.bg_color]
            block.bg_color.x = parseInt(block.x);
            block.bg_color.y = parseInt(block.y);

            this.fg_colors.push(block.fg_color);
            this.bg_colors.push(block.bg_color);
            this.blocks.push(block);
        }
    }

    getBlock(col, row){
        var index = (row * this.cols) + col;
        return this.blocks[index];
    }
}