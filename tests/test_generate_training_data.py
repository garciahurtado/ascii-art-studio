import unittest
import sys
import os

# Add project root to the Python path
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, proj_root)
sys.path.append(os.path.join(proj_root, 'lib'))

from const import INK_RED, INK_GREEN
from debugger import printc

from lib.cvtools.size_tools import adjust_img_size

# De-duplicated list of dimension pairs from the image comparison script
# Format: (in_width, in_height, expected_out_width, expected_out_height)
DIMENSION_TEST_CASES = [
    (128, 208, 328, 528),
    (244, 360, 360, 528),
    (324, 496, 344, 528),
    (360, 512, 376, 528),
    (486, 536, 384, 424),
    (500, 301, 528, 320),
    (500, 500, 384, 384),
    (512, 512, 384, 384),
    (608, 848, 384, 528),
    (680, 608, 432, 384),
    (816, 536, 528, 352),
    (1792, 752, 528, 224),
    (1792, 768, 528, 232),
    (1920, 784, 528, 216),
    (1920, 800, 528, 224),
    (1920, 904, 528, 248),
]

class TestAdjustImgSize(unittest.TestCase):

    def setUp(self):
        """Set up common parameters for tests (same dims as in training script)."""
        self.min_width = 384
        self.min_height = 384
        self.max_width = 528
        self.max_height = 528

    def test_adjust_img_size_with_real_data(self):
        """Test the adjust_img_size function with a list of real-world dimension pairs."""
        for in_w, in_h, expected_w, expected_h in DIMENSION_TEST_CASES:
            with self.subTest(in_w=in_w, in_h=in_h):
                new_width, new_height = adjust_img_size(
                    in_w, in_h,
                    self.min_width, self.min_height,
                    self.max_width, self.max_height
                )
                self.assertEqual((new_width, new_height), (expected_w, expected_h))

        printc(f"Tested {len(DIMENSION_TEST_CASES)} dimension pairs", INK_GREEN)

if __name__ == '__main__':
    unittest.main()
