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
        """Initializes the encoder with output dimensions and file path.

        Args:
            resolution (tuple): The (width, height) of the output video in pixels.
            char_size (tuple): The (width, height) of a single character block in pixels.
            output_file (str): The path to the target binary file.
        """
        (self.width, self.height) = resolution
        (self.char_width, self.char_height) = char_size
        self.output_file = output_file

    def export_binary_header(self, num_frames, duration, fps, charset_name):
        """Writes the CPEG stream header to the output file.

        The header contains global metadata for the video stream. It is written once
        at the beginning of the file.

        Binary Layout:
        - Bytes 1-4:   Magic number ('CPEG')
        - Byte 5:      FPS
        - Bytes 6-8:   Total number of frames (3 bytes)
        - Bytes 9-10:  Duration in seconds (unsigned short)
        - Bytes 11-12: Frame width in pixels (unsigned short)
        - Bytes 13-14: Frame height in pixels (unsigned short)
        - Byte 15:     Character width in pixels
        - Byte 16:     Character height in pixels
        - Byte 17:     Length of charset name
        - Bytes 18+:   Charset name (ASCII)
        """
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
        """Determines whether to export a full or differential frame to the binary encoder.

        If a diff frame is requested, this method first calculates the changed blocks.
        If the number of changes exceeds a threshold (currently 2/3 of total blocks),
        it will revert to exporting a full frame to reduce file size, since diff frames are 50% larger.
        """

        if is_full:
            flattened = current_char_map.flatten()
        else:
            diff_char_map = self.get_diff_char_map(last_char_map, current_char_map)
            threshold = current_char_map.shape[0] * current_char_map.shape[1] * (4 / 6)

            """ Force a full frame if the number of changed blocks exceeds the threshold """
            if len(diff_char_map) > threshold:
                is_full = True
                flattened = current_char_map.flatten()
            else:
                flattened = diff_char_map

        self.export_binary_frame(flattened, is_full)

    def export_binary_frame(self, blocks, is_full=True):
        """
        Serializes a frame's character and color data into the CPEG binary format.

        This method handles two distinct frame types:
        - Full frames (is_full=True): Writes the entire character map. This is used to describe keyframes or when the delta
        from the previous frame is too large.
        - Diff frames (is_full=False): Writes only the blocks that have changed since the last frame, including their
        absolute positions. This is standard practice in movie encoding to conserve space (P-frames).

        The method constructs a binary header containing frame metadata (type, dimensions) followed by the packed frame
        data for each character block. It directly appends the resulting binary data to the output file, so it's
        a linear and order-dependent format.

        The binary header is packed as follows:
        - Byte 1: Frame type (1 for Full, 2 for Diff)
        - Byte 2: Frame width in character blocks (0-255)
        - Byte 3: Frame height in character blocks (0-255)
        - Bytes 4-5: Number of character blocks included in this frame (unsigned short).

        FUTURE:
        - block size, in bits / color format (to help with different color resolutions)
        - FPS
        - Add a stream header section, rather than frame header, to avoid repetition
        - Character set name / ID
        """
        num_blocks = len(blocks)
        block_width, block_height = int(self.width / self.char_width), int(self.height / self.char_height)

        flags = self.FRAME_FULL if is_full else self.FRAME_DIFF

        header = struct.pack(
            '<cccH',
            self.as_bytes(flags),
            self.as_bytes(block_width),
            self.as_bytes(block_height),
            num_blocks)

        all_data = header

        for array_block in blocks.flatten():
            char = array_block.character
            if flags == self.FRAME_DIFF:
                # Absolute character (specifies position)
                block = self.pack_block(char.index, array_block.fg_color, array_block.bg_color, array_block.pos['x'],
                                        array_block.pos['y'])
            elif flags == self.FRAME_FULL:
                # Relative character (placed next to previous one)
                block = self.pack_block(char.index, array_block.fg_color, array_block.bg_color)

            all_data += block

        with open(self.output_file, 'ab') as file:
            file.write(all_data)

    def embed_colors(self, charmap, fg_colors_idx, bg_colors_idx):
        """
        Embeds the foreground and background colors of each block within the character inside the charmap
        :param charmap: List of ASCII characters representing the image, as a 2D array
        :param fg_colors_idx: List of indices of foreground colors to go with the charmap
        :param bg_colors_idx: List of indices of background colors to go with the charmap
        :return:
        """
        for [block, fg_color, bg_color] in zip(charmap.flatten(), fg_colors_idx, bg_colors_idx):
            block.fg_color = fg_color[0]
            block.bg_color = bg_color[0]

        return charmap

    def get_diff_char_map(self, previous_frame, current_frame):
        """Compares two frames and returns a list of changed blocks.

        A block is considered changed if its character index, foreground, or
        background color differs. Changed blocks are annotated with their
        absolute (x, y) position.
        """
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

    def export_ascii_data(self, blocks, fg_colors, bg_colors):
        """Exports frame data to a text file for web-based debugging.

        Serializes frame dimensions, char indices, and color data into a
        comma-separated string, writing the output to 'web/frame_data.txt'.
        """
        
        height = blocks.shape[0]
        width = blocks.shape[1]
        blocks = blocks.flatten()
        fg_colors = fg_colors.reshape(-1, 3)
        bg_colors = bg_colors.reshape(-1, 3)

        col = 0
        all_data = f'dim={width},{height},,'

        for [block, fg_color, bg_color] in zip(blocks, fg_colors, bg_colors):
            data = []
            char = block.character
            data.append(str(char.index))
            data.append(np.array2string(fg_color, separator=',').replace(" ", ""))
            data.append(np.array2string(bg_color, separator=',').replace(" ", ""))

            col += 1

            all_data += ','.join(data)

        with open('web/frame_data.txt', 'w') as file:
            file.write(all_data + '|')

    """------------- Utility functions ---------------------------------------------------------------------"""

    def pack_block(self, char_index, fg_color, bg_color, x=None, y=None):
        """Packs a character block into a binary format.

        Handles two formats:
        - Relative (4 bytes): Used for full frames. Encodes char index, fg color, bg color.
        - Absolute (6 bytes): Used for diff frames. Adds x, y coordinates.
        """

        # Convert from long to short (2 bytes)
        char_index = np.ushort(int(char_index))
        fg_color = np.ushort(int(fg_color))
        bg_color = np.ushort(int(bg_color))

        if (x is None) or (y is None):
            # Relative block (position is implicit)
            block = struct.pack('<Hcc', char_index, self.as_bytes(fg_color), self.as_bytes(bg_color))
        else:
            # Absolute block (position is explicit)
            block = struct.pack('<Hcccc', char_index, self.as_bytes(fg_color), self.as_bytes(bg_color),
                                self.as_bytes(x), self.as_bytes(y))

        return block

    def pack_color_block(self, red, green, blue):
        """Packs an RGB color into a 3-byte binary format."""
        block = struct.pack('<ccc', self.as_bytes(red), self.as_bytes(green), self.as_bytes(blue))
        return block

    def as_bytes(self, value, num_bytes=1):
        """Converts an integer or character to a little-endian byte string.

        Args:
            value (int or str): The value to convert. If a string, its ordinal value is used.
            num_bytes (int): The number of bytes to use in the output
        """
        if type(value) is str:
            value = ord(value)

        value = int(value)
        bytes = value.to_bytes(num_bytes, 'little')
        return bytes
