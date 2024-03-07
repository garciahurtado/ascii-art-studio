import numpy as np

from .palette import Palette
import cv2 as cv

class PaletteExtractor:
  def extract_palette(self, image, num_colors=128, previous_palette=None):
    points = np.array(self.get_colors(image), dtype=np.float32)

    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_MAX_ITER, 1, 10)

    # Reuse the centers from the last kmeans run, if available, to minimize time to convergence
    if previous_palette is not None:
      centers = previous_palette.colors
    else:
      centers = None

    compactness, color_idx, rgbs = cv.kmeans(points, num_colors, None, criteria, 10, cv.KMEANS_PP_CENTERS, centers)

    palette = Palette(rgbs)
    palette.color_idx = color_idx

    return color_idx, palette


  def get_colors(self, image):
    colors = []
    width, height = image.shape[0], image.shape[1]

    for x in range(0, width):
      for y in range(0, height):
        pixel = image[x, y]
        colors.append(pixel)

    return colors
