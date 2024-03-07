'''Basic keyboard shortcuts for a OpenCV image / video display window.

Requires OpenCV and Numpy

Built in Shortcuts:

s - Take screenshot and save it in 'resources/screenshots' as a PNG
ESC / q - quit application / terminate window (if used with looper.py

'''

import cv2 as cv
import numpy as np
from datetime import datetime
from ui import looper

class Window():
    def __init__(self, window_name='image'):
        self.window_name = window_name
        self.last_image = None
        self.show_fps = False
        self.ui_color = (0,255,0)
        self.ui_text = {}
        self.ui_font = cv.FONT_HERSHEY_PLAIN
        cv.namedWindow(window_name)

    def show(self, img):
        if self.show_fps:
            fps = f'{looper.actual_fps:.2f}'
            avg_fps = f'{looper.avg_fps:.2f}'

            x, y = 10, 20
            cv.putText(img, f'FPS: {avg_fps} ({fps})', (x, y), self.ui_font, 1, self.ui_color, 2)

            # Custom UI text
            y = 40
            for key, value in self.ui_text.items():
                cv.putText(img, f'{key}: {value}', (x, y), self.ui_font, 1, self.ui_color, 2)
                y += 20

        self.render_img(img)
        self.last_image = img


    def render_img(self, img):
        cv.imshow(self.window_name, img)

    def get_key(self):
        key = self.capture_key()

        # 's' to take a screenshot of the last image shown in this window
        if key == ord('s'):
            self.take_screenshot()

        elif key == 27:
            return looper.BREAK

        return key


    def take_screenshot(self):
        if self.last_image is not None:
            capture = self.last_image.copy()

            # Render a faux flash as a visual indicator of screen capture
            flash = np.full_like(self.last_image, (255, 255, 255), np.uint8)
            self.show(flash)
            cv.waitKey(200)  # Wait 200ms, shows "flash" for a moment and forces redraw
            filename = f"resources/screenshots/capture_{datetime.now():%Y-%m-%d-%H%M%S}.png"
            cv.imwrite(filename, capture)
            print(f"Saved screenshot to {filename}")


    def capture_key(self):
       return cv.waitKeyEx(1)