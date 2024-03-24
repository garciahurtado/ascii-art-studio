class PaletteFrame {
    constructor() {
        this.blocks;
        this.next_frame;
        this.prev_frame;
        this.colors = [];
    }

    setBlocks(blocks){
        this.blocks = blocks;
        for(var i in blocks){
            var block = blocks[i];
            this.colors.push([block.red,block.green,block.blue]);
        }
    }

    getBlock(col, row){
        var index = (row * this.cols) + col;
        return this.blocks[index];
    }
}