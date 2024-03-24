window.onload = function() {
    var el=document.getElementById("player");
    var ctx=el.getContext('2d');
    var char_width = 8;
    var char_height = 8;
    var player = new AsciiMoviePlayer(ctx, "charpeg/video_stream.cpeg" , char_width, char_height);

    player.play_button = document.getElementById("play_button");
    player.pause_button = document.getElementById("pause_button");

    var start_time = performance.now();
    var start_frame_time = performance.now();
    var end_time =   performance.now();

    var all_chars = [];


    charset = document.getElementById("charset");
    charset.crossOrigin = "Anonymous";
    charset.src = "img/charsets/amstrad-cpc.png"; // can also be a remote URL e.g. http://

    player.decoder = new VideoDecoder();
    player.decoder.progress_el = document.getElementById("prog_bar");

    var paletteLayer = document.getElementById("palette");

    // Chop up the charset into characters, and create a bitmap for each
    charset.onload = function() {
        player.loadCharset(charset, char_width, char_height);
    };

    // Wire up controls
    document.getElementById("pause").onclick = function(){
        player.pause();
    };

    document.getElementById("start").onclick = function(){
        player.reset();
        player.play();
    };

    document.getElementById("end").onclick = function(){
        player.pause();
        player.goToFrame(player.total_frames - 1);
    };

    document.getElementById("prev_frame").onclick = function(){
        player.stop();
        player.goToPrevFrame();
    };

    document.getElementById("next_frame").onclick = function(){
        player.stop();
        player.goToNextFrame();
    };

    document.getElementById("view_grid").onclick = function(){
        if(player.show_grid){
            player.show_grid = false;
        } else {
            player.show_grid = true;
        }
        player.showFrame(player.runner.getCurrentFrame());
    };

    document.getElementById("show_palette").onclick = function(){
        if(player.show_palette){
            player.show_palette = false;
            paletteLayer.style.visibility = 'hidden';
        } else {
            player.show_palette = true;
            paletteLayer.style.visibility = 'visible';
        }
        player.showFrame(player.runner.getCurrentFrame());
    };

    var scrub_bar = document.getElementById("scrub_bar");
    scrub_bar.onclick = function(e){
        pos = getCursorPosition(prog_bar, e);
        player.seekToPosition(pos.x);
    };
};

function getCursorPosition(el, event) {
    const rect = el.getBoundingClientRect()
    const x = event.clientX - rect.left
    const y = event.clientY - rect.top
    return {x: x, y: y};
}