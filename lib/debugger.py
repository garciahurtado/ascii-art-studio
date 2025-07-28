from const import *

def printb(message):
    """ Print a string and then print a series of backspaces equal to its length, to implement statically positioned
     text in the terminal """
    num_back = len(message)
    eraser = '\b' * num_back
    print(message, end='')
    print(eraser, end='')

def printc(message, color=INK_RED, newline=True):
    if newline:
        print(f"{color}{message}", INK_END)
    else:
        print(f"{color}{message}", INK_END, end="")