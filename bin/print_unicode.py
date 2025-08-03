import json

from charset import Charset


def do_main():
    charset = Charset(8,16)
    charset.load("ubuntu-mono_8x16.png")
    charset.print_unicode_chars()


do_main()