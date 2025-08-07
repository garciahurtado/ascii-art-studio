# ANSI colors
from os.path import dirname, abspath, join

# Standard Colors
INK_BLACK = "\033[30m"          # Black (standard)
INK_RED = "\033[31m"            # Red (standard)
INK_GREEN = "\033[32m"          # Green (standard)
INK_YELLOW = "\033[33m"         # Yellow (standard)
INK_BLUE = "\033[34m"           # Blue (standard)
INK_MAGENTA = "\033[35m"        # Magenta (standard)
INK_CYAN = "\033[36m"           # Cyan (standard)
INK_WHITE = "\033[37m"          # White (standard)

# Bright/Bold Colors
INK_GRAY = "\033[90m"           # Bright Black (Gray)
INK_BRIGHT_RED = "\033[91m"     # Bright Red
INK_BRIGHT_GREEN = "\033[92m"   # Bright Green
INK_BRIGHT_YELLOW = "\033[93m"  # Bright Yellow
INK_BRIGHT_BLUE = "\033[94m"    # Bright Blue
INK_BRIGHT_MAGENTA = "\033[95m" # Bright Magenta (often used as HEADER)
INK_BRIGHT_CYAN = "\033[96m"    # Bright Cyan
INK_BRIGHT_WHITE = "\033[97m"   # Bright White

# Special Formatting
INK_BOLD = "\033[1m"            # Bold text
INK_UNDERLINE = "\033[4m"       # Underlined text
INK_REVERSE = "\033[7m"         # Reversed (background/foreground swap)

# Reset
INK_END = "\033[0m"             # Reset all styles (simpler than "\033[0;0m")

# Paths
PROJECT_ROOT = dirname(abspath(join(dirname(__file__))))
DATASETS_ROOT = join(PROJECT_ROOT, 'datasets')

print(f"PROJECT_ROOT: {PROJECT_ROOT}")
