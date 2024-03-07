class Block:
    character = None
    fg_color = None
    bg_color = None
    x = None
    y = None
    char_index = None

    def __init__(self, character):
        self.character = character
        self.char_index = character.index

