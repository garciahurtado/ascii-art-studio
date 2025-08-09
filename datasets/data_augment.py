import math
import os

import argparse
import numpy as np
from PIL import Image

from cvtools import size_tools
from cvtools.size_tools import Dimensions
from const import out_min_width, out_min_height, out_max_width, out_max_height

out_min_dims = Dimensions(out_min_width, out_min_height)
out_max_dims = Dimensions(out_max_width, out_max_height)


def create_augmented_images(input_dir, output_dir):
    """
    Generates a new, augmented dataset by shifting source images.
    returns: number of images created
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    new_img_count = 0
    input_dir = os.path.realpath(input_dir)
    shifts = [
        Dimensions(4, 0),
        Dimensions(-4, 0),
        Dimensions(0, 4),
        Dimensions(0, -4)]

    for image_name in os.listdir(input_dir):
        if not image_name.endswith(".png"):
            continue

        source_image = Image.open(os.path.join(input_dir, image_name))
        source_dims = Dimensions(source_image.size[0], source_image.size[1])

        # Step 1: Figure out final, resized dimensions
        out_dims = size_tools.adjust_img_size(source_dims, out_min_dims, out_max_dims)
        out_dims = Dimensions(out_dims[0], out_dims[1])
        resize_ratio = source_dims.width / out_dims.width

        # Now that we know the final output size, we can work backwards to decide how much of a shift to apply:
        # - out_shift: the desired shift (in pixels) of the resized image
        # - orig_shift: the shift we will have to apply to the original image in order to get out_shift after resizing
        final_image_name = image_name[:-4]  # remove the .png

        for shift_dims in shifts:
            final_image = shift_image(source_image, shift_dims, resize_ratio)

            # Save the new augmented image
            px = f"{shift_dims.width:02d}x{shift_dims.height:02d}"
            output_filename = f"{final_image_name}_shifted_{px}.png"
            final_image.save(os.path.join(output_dir, output_filename))
            print(f'> {output_filename} created')

            new_img_count += 1

    return new_img_count


def shift_image(source_image, shift_dims: Dimensions, resize_ratio=1.0):
    source_dims = Dimensions(source_image.size[0], source_image.size[1])
    orig_shift = Dimensions(
        round(shift_dims.width * resize_ratio),
        round(shift_dims.height * resize_ratio))

    print(
        f"Will shift original by {orig_shift.width}x{orig_shift.height} to get {shift_dims.width}x{shift_dims.height} in the output image")

    # Create a new blank canvas, the same size as the original image
    final_image = Image.new('RGB', (source_dims.width, source_dims.height))

    # Paste the original image on top, but shifted
    final_image.paste(source_image, (orig_shift.width, orig_shift.height))
    return final_image


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, default=current_dir)
    parser.add_argument("output_dir", type=str, default="augmented")
    parser.add_argument("--variants_per_image", type=int, default=2)

    args = parser.parse_args()
    create_augmented_images(args.input_dir, args.output_dir, args.variants_per_image)
