import os
import time
import math
import numpy as np

# User libs
from ascii.ascii_converter import AsciiConverter
from charset import Charset
from cvtools import size_tools as tools
from ascii.block import Block
import torch

from net.ascii_classifier_network import AsciiClassifierNetwork
from pytorch import model_manager

class NeuralAsciiConverterPytorch(AsciiConverter):
    model: AsciiClassifierNetwork = None

    def __init__(self, charset: Charset, model_filename, charsize, num_labels=None, model_subdir=None):
        self.charset: Charset = charset
        self.char_width, self.char_height = charsize

        # Index characters by their position in which they were loaded into the charset
        self.charmap = {character.index: character for character in charset}
        self.load_model(model_subdir, model_filename, num_labels)

    def load_model(self, model_subdir, filename, num_labels):
        # Load ML model source code and pretrained weights from model file
        self.model = model_manager.load_model(model_subdir, filename, num_labels)
        self.model = self.model.cuda()
        return self.model

    def convert_image(self, input_image):
        start = time.time()

        width, height = input_image.shape[1], input_image.shape[0]

        char_height, char_width = self.char_height, self.char_width
        block_cols, block_rows = math.ceil(width / char_width), math.ceil(height / char_height)

        self.used_chars = []  # Keep track of the unique characters used in this rendering, for analytics
        self.match_char_map = np.full((block_rows, block_cols), 0, dtype=object)
        self.candidate_chars = [[[] for i in range(block_cols)] for j in range(block_rows)]

        output_image = input_image.copy()

        # Slice up the B&W input image into blocks and match each of them to an ASCII characters
        blocks = self.convert_to_blocks(input_image)
        blocks_gpu = torch.from_numpy(blocks)
        blocks_gpu = blocks_gpu.permute(0, 3, 1, 2) # Reshape input tensor to match [batch, channels, width, height]
        blocks_gpu = blocks_gpu.cuda()
        predictions = self.predict_blocks(self.model, blocks_gpu)

        # Retrieve the characters referenced by the labels returned by the model
        characters = self.predictions_to_characters(predictions)
        characters = self.set_full_empty_chars(characters, blocks)
        self.used_chars = characters

        char_index = 0

        # Take the list of predicted characters and copy their images into the output images
        for row in range(0, block_rows):
            y = row * self.char_height

            for col in range(0, block_cols):
                x = col * self.char_width
                character = characters[char_index]
                output_image[y:y + char_height, x:x + char_width] = character.img
                self.candidate_chars[row][col] = [character]
                new_block = Block(character)
                self.match_char_map[row][col] = new_block

                char_index += 1

        # Return the list of Character objects used, along with the output image
        end = time.time()
        # print("time elapsed {} milli seconds".format((end - start) * 1000))

        return output_image

    def predictions_to_characters(self, predictions):
        ascii = torch.argmax(predictions, dim=-1)
        ascii = ascii.cpu()

        chars = []
        ascii = ascii.numpy()
        for code in ascii:
            chars.append(self.charmap[code])

        return chars

    def predict_blocks(self, model, images):
        self.model.eval()

        return model(images)

    def convert_to_blocks(self, input_image):
        """Slice up the B&W input image into blocks and reshape into required format"""

        img_blocks = tools.as_blocks(input_image, (self.char_height, self.char_width))
        img_blocks = img_blocks.astype(np.float32)
        img_blocks = img_blocks.reshape(-1, self.char_height, self.char_width, 1)

        return img_blocks

    def set_full_empty_chars(self, charlist, images):
        charlist_out = []
        threshold = 2

        for char, block in zip(charlist, images):
            if (tools.is_almost_full(block, threshold)):
                charlist_out.append(self.charset.full_char)
            elif (tools.is_almost_empty(block, threshold)):
                charlist_out.append((self.charset.empty_char))
            else:
                charlist_out.append(char)

        return charlist_out

