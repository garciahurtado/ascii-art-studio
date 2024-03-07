'''Provides a decorator to run functions in a loop at a maximum FPS.

This module provides a decorator which will run the decorated function
in a loop, with a maximum FPS limit.

Usage:

import looper

@looper.fps(30)
def your_function(your_arguments):
    # Function definition

    if(condition): return looper.BREAK

This will run your function in a loop at a *maximum* of 30 FPS. Note that there
is no guarantee of minimum FPS, so your function could run at lower than 30 FPS,
depending on its performance.

In order to break out of the loop, simply have your function return "looper.BREAK"

Since this module keeps track of the last frame timestamp in a pseudo-global static
variable, it can only be used once per Python process.

'''

import sys
import time

# instance of the current module object, in order to keep "Global" vars
import numpy as np

this = sys.modules[__name__]
this.time_last_frame = 0
this.actual_fps = 0
this.fps_history = []
this.avg_fps = 0
this.time_elapsed = 0
BREAK = 9999

def fps(fps):
    def inner_decorator(func): # Inner decorator is needed in order to use decorator arguments (ie: FPS)
        def wrapper(*args, **kwargs):
            while True:
                time_now = time.time()

                if this.time_last_frame > 0:
                    this.time_elapsed = time_now - this.time_last_frame
                else:
                    this.time_elapsed = 0

                if(this.time_elapsed):
                    this.actual_fps = 1 / this.time_elapsed
                    this.fps_history.append(this.actual_fps)
                    fps_len = len(this.fps_history)

                    samples = 20
                    if fps_len > samples: # Truncate the list of FPS samples
                        this.fps_history = this.fps_history[fps_len-samples:fps_len]

                    history = sorted(this.fps_history)[1:-1] # Eliminate outliers
                    this.avg_fps = np.average(history)
                else:
                    this.actual_fps = 0

                # More time has passed since the last frame, than: 1 second divided by the FPS, so
                # run the wrapped function
                if ((this.time_last_frame == 0) or (this.time_elapsed > 1. / fps)):
                    this.time_last_frame = time.time()

                    ret = func(*args, **kwargs)

                    if(ret == BREAK):
                        break
            return ret
        return wrapper
    return inner_decorator