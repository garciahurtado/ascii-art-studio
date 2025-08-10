import time
from functools import wraps
from collections import defaultdict
from typing import Dict, List

from const import INK_GREEN, INK_CYAN, INK_YELLOW
from debugger import printc


class Profiler:
    _instance = None
    _timing_data: Dict[str, List[float]] = defaultdict(list)
    _call_counts: Dict[str, int] = defaultdict(int)
    _enabled = True

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Profiler, cls).__new__(cls)
        return cls._instance

    def profile(self, func):
        """Decorator to profile function execution time."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self._enabled:
                return func(*args, **kwargs)
            printc(f"Profiling {func.__qualname__}", INK_GREEN)  # Debug print
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start_time

            self._timing_data[func.__qualname__].append(elapsed)
            self._call_counts[func.__qualname__] += 1
            printc(f"Profiled {func.__qualname__} in {elapsed:.6f}s", INK_CYAN)  # Debug print

            return result

        return wrapper

    @classmethod
    def print_report(cls, sort_by: str = 'total'):
        """Print a summary report of all profiled functions.

        Args:
            sort_by: How to sort the results ('total', 'average', 'calls')
        """
        if not cls._timing_data:
            printc("No profiling data available!!")
            return

        # Calculate statistics
        func_stats = []
        for func_name, times in cls._timing_data.items():
            total = sum(times)
            avg = total / len(times)
            func_stats.append((
                func_name,
                len(times),  # calls
                total,  # total time
                avg,  # average time
                min(times),  # min time
                max(times)  # max time
            ))

        # Sort based on the specified column
        sort_columns = {
            'calls': 1,
            'total': 2,
            'average': 3,
            'min': 4,
            'max': 5
        }
        sort_idx = sort_columns.get(sort_by, 2)  # Default to sorting by total time
        func_stats.sort(key=lambda x: x[sort_idx], reverse=True)

        # Print the report
        print("\n" + "=" * 120)
        print(f"{'Function':<50} {'Calls':>10} {'Total (s)':>12} {'Avg (s)':>12} {'Min (s)':>12} {'Max (s)':>12}")
        print("-" * 120)

        for name, calls, total, avg, min_t, max_t in func_stats:
            print(f"{name:<50} {calls:>10} {total:>12.6f} {avg:>12.6f} {min_t:>12.6f} {max_t:>12.6f}")

        print("=" * 120 + "\n")

    @classmethod
    def reset(cls):
        """Clear all profiling data."""
        cls._timing_data.clear()
        cls._call_counts.clear()

    @classmethod
    def enable(cls):
        """Enable profiling."""
        cls._enabled = True

    @classmethod
    def disable(cls):
        """Disable profiling (useful for production)."""
        cls._enabled = False

    @classmethod
    def debug_info(cls):
        """Print debug information about the profiler state."""
        printc("\n=== Profiler Debug Info ===", INK_YELLOW)
        printc(f"Enabled: {cls._enabled}", INK_YELLOW)
        printc(f"Profiled functions: {list(cls._timing_data.keys())}", INK_YELLOW)
        printc(f"Call counts: {dict(cls._call_counts)}", INK_YELLOW)
        printc("=========================\n", INK_YELLOW)


# Create a global instance for easy use
profiler = Profiler()

"""
Example usage:
if __name__ == "__main__":
    @profiler.profile
    def example_function(n):
        total = 0
        for i in range(n):
            total += i
        return total


    # Call some functions
    for _ in range(5):
        example_function(1000000)

    # Print the report
    Profiler.print_report(sort_by='total')
"""
