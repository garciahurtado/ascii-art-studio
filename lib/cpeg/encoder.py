import struct
import numpy as np


class Encoder():
    '''Frame types'''
    FRAME_FULL = 1
    FRAME_DIFF = 2
    FRAME_PALETTE = 3

    '''A CPEG movie encoder that uses frame data in the form of RGB palettes, and indexed character frames,
    and encodes it into the binary format defined by CPEG'''
    def __init__(self, resolution, char_size, output_file):
        (self.width, self.height) = resolution
        (self.char_width, self.char_height) = char_size
        self.output_file = output_file


    def export_binary_header(self, num_frames, duration, fps, charset_name):
        '''Exports the first frame of the stream, which contains global data, such as magic number,
        resolution, FPS, character dimensions, total number of frames and charset used.'''

        '''
        Bytes:
        1,2,3,4: magic number (CPEG in ASCII)
        5: FPS
        6,7,8: Total number of frames
        5,6: duration, in seconds
        7: frame width, in pixels
        8,9: frame height, in pixels
        10,11: char width, in pixels
        10: char height, in pixels
        11: Charset name (length)
        12+: Charset name
        '''

        magic_number = struct.pack('<cccc', *[self.as_bytes(char) for char in 'CPEG'])
        all_data = magic_number

        fps = struct.pack('<c', self.as_bytes(fps))
        all_data += fps

        num_frames = struct.pack("<L", num_frames)
        all_data += num_frames[:3]

        duration = struct.pack('<H', np.ushort(duration))
        all_data += duration

        width = struct.pack('<H', np.ushort(self.width))
        all_data += width

        height = struct.pack('<H', np.ushort(self.height))
        all_data += height

        char_width = struct.pack('<c', self.as_bytes(self.char_width))
        all_data += char_width

        char_height = struct.pack('<c', self.as_bytes(self.char_height))
        all_data += char_height

        charset_name_length = struct.pack('<c', self.as_bytes(len(charset_name)))
        all_data += charset_name_length

        format_str = 'c' * len(charset_name)
        charset_name = struct.pack('<' + format_str, *[self.as_bytes(char) for char in charset_name])
        all_data += charset_name

        with open(self.output_file, 'wb') as file:
            file.write(all_data)


    def export_binary_palette(self, palette):
        '''Exports a color palette to the ASCII video frame format'''
        flags = 3

        # Frame Header Byte #:
        # 1: Frame type:
        # 2,3: padding so that the header will be the same size as normal frame headers
        # 4,5: number of color blocks
        #
        num_colors = len(palette.colors)
        header = struct.pack('<cHH', self.as_bytes(flags), np.ushort(0), np.ushort(num_colors))
        all_data = header

        for color in palette.colors:
            block = self.pack_color_block(color[0], color[1], color[2])
            all_data += block

        with open(self.output_file, 'ab') as file:
            file.write(all_data)


    def export_full_or_diff_frame(self, current_char_map, last_char_map, is_full=True):
        if is_full:
            flattened = current_char_map.flatten()
        else:
            diff_char_map = self.get_diff_char_map(last_char_map, current_char_map)

            # If the diff contains too many blocks, it may result in a small binary size to just do a full frame
            # Diff blocks are 50% larger than full blocks
            threshold = current_char_map.shape[0] * current_char_map.shape[1] * (4 / 6)

            if (len(diff_char_map) > threshold):
                # This is the branch where we should introduce a palette change
                is_full = True
                flattened = current_char_map.flatten()
            else:
                flattened = diff_char_map

        self.export_binary_frame(flattened, is_full)


    def export_binary_frame(self, chars, is_full=True):
        '''Exports a single video frame to the specified binary format '''
        num_chars = len(chars)
        block_width, block_height = int(self.width / self.char_width), int(self.height / self.char_height)

        flags = self.FRAME_FULL if is_full else self.FRAME_DIFF

        # Header Byte #:
        # 1: Frame type / flags
        # 2: frame width, in blocks(0 - 255)
        # 3: frame height, in blocks(0 - 255)
        # 4, 5: number of blocks
        #
        # FUTURE:
        # - block size, in bits / color format (to help with different color resolutions)
        # - FPS
        # - Add a stream header section, rather than frame header, to avoid repetition
        # - Character set name / ID
        header = struct.pack(
            '<cccH',
            self.as_bytes(flags),
            self.as_bytes(block_width),
            self.as_bytes(block_height),
            num_chars)

        all_data = header

        for char in chars:
            if flags == self.FRAME_DIFF:
                # Absolute character (specifies position)
                block = self.pack_block(char.char_index, char.fg_color, char.bg_color, char.pos['x'], char.pos['y'])
            elif flags == self.FRAME_FULL:
                # Relative character (placed next to previous one)
                block = self.pack_block(char.char_index, char.fg_color, char.bg_color)

            all_data += block

        with open(self.output_file, 'ab') as file:
            file.write(all_data)

    def embed_colors(self, charmap, fg_colors_idx, bg_colors_idx):
        for [char, fg_color, bg_color] in zip(charmap.flatten(), fg_colors_idx, bg_colors_idx):
            char.fg_color = fg_color[0]
            char.bg_color = bg_color[0]

        return charmap

    def get_diff_char_map(self, previous_frame, current_frame):
        ''' Compares two full CPEG encoded frames block by block, and returns a list
         of those blocks that differ either in character code, FG or BG color '''
        diff_blocks = []

        for y, row in enumerate(previous_frame):
            for x, last_char in enumerate(row):
                last_color_fg = last_char.fg_color
                last_color_bg = last_char.bg_color

                new_char = current_frame[y][x]
                new_color_fg = new_char.fg_color
                new_color_bg = new_char.bg_color

                char_match = last_char.char_index == new_char.char_index
                fg_color_match = (last_color_fg == new_color_fg)
                bg_color_match = (last_color_bg == new_color_bg)

                if (not char_match) or \
                        (not fg_color_match) or \
                        (not bg_color_match):
                    new_char.pos = {}
                    new_char.pos['x'] = x
                    new_char.pos['y'] = y
                    diff_blocks.append(new_char)

        return diff_blocks


    def export_ascii_data(self, chars, fg_colors, bg_colors):
        height = chars.shape[0]
        width = chars.shape[1]
        chars = chars.flatten()
        fg_colors = fg_colors.reshape(-1, 3)
        bg_colors = bg_colors.reshape(-1, 3)

        col = 0
        all_data = f'dim={width},{height},,'

        for [char, fg_color, bg_color] in zip(chars, fg_colors, bg_colors):
            data = []
            data.append(str(char.index))
            data.append(np.array2string(fg_color, separator=',').replace(" ", ""))
            data.append(np.array2string(bg_color, separator=',').replace(" ", ""))

            col += 1

            all_data += ','.join(data)

        with open('web/frame_data.txt', 'w') as file:
            file.write(all_data + '|')

    """------------- Utility functions ---------------------------------------------------------------------"""

    def pack_block(self, char_index, fg_color, bg_color, x=None, y=None):
        ''' Packs a single ASCII block (character index, foreground color, background color)
        into a series of bytes 4 bytes'''

        # Convert from long to short (2 bytes)
        char_index = np.ushort(int(char_index))
        fg_color = np.ushort(int(fg_color))
        bg_color = np.ushort(int(bg_color))

        if (x is None) or (y is None):
            # Absolute block
            block = struct.pack('<Hcc', char_index, self.as_bytes(fg_color), self.as_bytes(bg_color))
        else:
            # Relative block
            block = struct.pack('<Hcccc', char_index, self.as_bytes(fg_color), self.as_bytes(bg_color), self.as_bytes(x), self.as_bytes(y))

        return block

    def pack_color_block(self, red, green, blue):
        block = struct.pack('<ccc', self.as_bytes(red), self.as_bytes(green), self.as_bytes(blue))
        return block

    def as_bytes(self, value, num_bytes=1):
        '''Return the bytes representation of the given value, using as many bytes as specified
        '''
        if type(value) is str:
            value = ord(value)

        value = int(value)
        bytes = value.to_bytes(num_bytes, 'little')
        return bytes


