/**
 * Executes a single function (a frame renderer) no more than once
 * per interval which defines the FPS required.
 */
class FpsRunner {
    constructor(fps){
        this.fps = fps;
        this.ms_per_frame = 1000 / fps;
        this.callback;
        this.time_last_frame = 0;
        this.running = false;
        this.frames;
        this.frame_index = 0;
        this.on_finish; // callback used when playback is finished with all frames
    }

    next(){
        var now = performance.now();
        var frame = this.getCurrentFrame();

        if((now - this.time_last_frame > this.ms_per_frame) && this.running){
            var start_time = performance.now();
            this.callback(frame);
            this.time_last_frame = performance.now();
            this.frame_index++;
        }

        if(this.frame_index >= this.frames.length){ // we've reached the end
            this.frame_index = this.frames.length - 1;
            this.stop();
            if(this.on_finish){
                this.on_finish();
            }
        } else {
            var self = this;
            requestAnimationFrame(function(){
                self.next();
            });
        }
    }

    getCurrentFrame(){
        return this.frames[this.frame_index];
    }

    play(){
        this.running = true;
        this.next();
    }

    stop(){
        this.running = false;
    }
}