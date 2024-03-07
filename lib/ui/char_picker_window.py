import math

from ui.opencv_window import Window
import cv2 as cv
import numpy as np

class CharPickerWindow(Window):
    """"""

    def __init__(self, window_name):
        super(CharPickerWindow, self).__init__(window_name)
        self.char_width = 8
        self.char_height = 8
        self.converter = None
        self.selected_block = None

        self.scale = 4
        self.match_cols = 48
        self.match_rows = 6

        self.block_width = self.char_width * self.scale
        self.block_height = self.char_height * self.scale

        self._selected_char = None
        self.current_candidates = None
        self.ui_window = None

        cv.setMouseCallback(window_name, self.mouse_click)

    def mouse_click(self, event, mouse_x, mouse_y, flags, param):

        if event == cv.EVENT_LBUTTONDOWN:
            # mouse_x = mouse_x - (mouse_x % char_width)
            # mouse_y = mouse_y - (mouse_y % char_width)

            sel_col = int(mouse_x / self.block_width)
            sel_row = int(mouse_y / self.block_height)
            selected_block = (sel_row, sel_col)

            print('selected char: ' + str(selected_block))


    def show(self, img):
        output = self.show_char_matches(img)

        if self.selected_char is not None:
            output = self.outline_selected_char(output)

        super().show(output)

    def show_char_matches(self, contrast_img):
        candidates = self.converter.candidate_chars
        matches = self.converter.match_char_map
        masks = self.converter.mask_map

        # Scores
        pixel_diff = self.converter.pixel_diff_map
        dist_eucl = self.converter.dist_eucl_map
        pixel_factor = self.converter.pixel_factor_map
        hog_factor = self.converter.hog_factor_map

        total_rows = self.match_rows + 1

        output = np.full((self.char_height * self.scale * total_rows, self.char_width * self.scale * self.match_cols),
                         self.match_cols, np.uint8)

        if(self.selected_block is None):
            return output

        row, col = self.selected_block
        x, y = col * self.char_height, row * self.char_width


        img_block = contrast_img[y:y + self.char_height, x:x + self.char_width]
        img_block = self.resize_to_block(img_block)
        img_block_width, img_block_height = img_block.shape[1], img_block.shape[0]
        output[0:img_block_width, 0:img_block_height] = img_block
        candidates_list = self.current_candidates = candidates[row][col].copy()

        # Trim the candidates list if it's too long
        if len(candidates_list) > (self.match_cols * self.match_rows):
            candidates_list = candidates_list[0:self.match_cols * self.match_rows]

        match_char = None

        # Show the final match
        if (matches.any()):
            match_char = matches[row][col]

            if (match_char):
                match_img = self.resize_to_block(match_char.img)
                output[0:img_block_width, img_block_width:img_block_width * 2] = match_img

        # Mask that selected the final match (if any)
        if masks is not None:
            mask = masks[row][col]
            if (mask is not None):
                mask_img = self.resize_to_block(mask)
            else:
                mask_img = np.full((img_block_height, img_block_width), 64, np.uint8)
        else:
            mask_img = np.full((img_block_height, img_block_width), 64, np.uint8)

        output[0:img_block_width, img_block_width * 2:img_block_width * 3] = mask_img

        # Show all candidate characters that were considered, arranged in rows and columns
        x = 0
        for i, char in enumerate(candidates_list):
            img = self.resize_to_block(char.img)

            start_x = (i % self.match_cols) * img_block_width
            start_y = (math.floor(i / self.match_cols) + 1) * img_block_height

            output[start_y:start_y + img_block_height, start_x:start_x + img_block_width] = img

        output = cv.cvtColor(output, cv.COLOR_GRAY2BGR)

        # Draw lines
        line_color = (0, 190, 0)
        block_size = self.char_width * 4

        # Horizontal grid lines
        for i in range(1, self.match_rows + 1):
            cv.line(output, (0, block_size * i), (output.shape[1], block_size * i), line_color, 1)

        # vertical grid lines
        for j in range(1, self.match_cols):
            cv.line(output, (j * block_size, 0), (j * block_size, (total_rows * block_size) + 2), line_color, 1)

        # Add coordinates
        text_color = (255, 0, 255)
        text = f'({col},{row})'
        cv.putText(output, text, ((img_block_width * 5), 14), cv.FONT_HERSHEY_PLAIN, 1, text_color)

        # Show calculated diffs / scores for character
        cv.putText(output, f'Pixel diff: {pixel_diff[row][col]}', org=((img_block_width * 8), 14), color=text_color,
                   fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=1)
        cv.putText(output, f'HOG dist: {dist_eucl[row][col]}', org=((img_block_width * 8), 30), color=text_color,
                   fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=1)

        cv.putText(output, f'Pixel factor: {pixel_factor[row][col]}', org=((img_block_width * 20), 14), color=text_color,
                   fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=1)
        cv.putText(output, f'HOG factor: {hog_factor[row][col]}', org=((img_block_width * 20), 30), color=text_color,
                   fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=1)

        return output


    def select_block(self, coords):
        self.selected_block = coords

    def resize_to_block(self, img):
        scale = self.scale

        out = cv.resize(img, (int(8 * scale), int(8 * scale)), interpolation=cv.INTER_NEAREST)
        return out


    def outline_selected_char(self, img):
        color = (0,0,255)
        from_x = self.selected_char[0] * self.block_width
        from_y = (self.selected_char[1] + 1) * self.block_width
        output = cv.rectangle(img,
                              (from_x, from_y),
                              (from_x + self.block_width, from_y + self.block_height), color, 2)

        return output

    @property
    def selected_char(self):
        return self._selected_char

    @selected_char.setter
    def selected_char(self, coords):
        self._selected_char = coords

        char_code = self.get_char_code_from_coords(coords)

        if self.ui_window is not None:
            self.ui_window.set_override_char(char_code)

    def key_left(self):
        if self.selected_char is None:
            self.selected_char = (0,0)
        else:
            col = self.selected_char[0] - 1
            row = self.selected_char[1]

            if col < 0:
                col = self.match_cols - 1 # Wrap around

            self.selected_char = (col, row)

    def key_right(self):
        if self.selected_char is None:
            self.selected_char = (0,0)
        else:
            col = self.selected_char[0] + 1
            row = self.selected_char[1]

            if col > self.match_cols - 1:
                col = 0 # Wrap around

            self.selected_char = (col, row)


    def key_up(self):
        if self.selected_char is None:
            self.selected_char = (0,0)
        else:
            col = self.selected_char[0]
            row = self.selected_char[1] - 1

            if row < 0:
                row = self.match_rows

            self.selected_char = (col, row)


    def key_down(self):
        if self.selected_char is None:
            self.selected_char = (0,0)
        else:
            col = self.selected_char[0]
            row = self.selected_char[1] + 1

            if row > self.match_rows - 1:
                row = 0

            self.selected_char = (col, row)

    def get_char_code_from_coords(self, coords):
        col, row = coords
        index = (row * self.match_cols) + col

        character = self.current_candidates[index]

        return character.index




