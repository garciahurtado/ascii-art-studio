'''OpenGL / pyGame display window which shows arbitrary images (from OpenCV or otherwise)
by using a textured quad at 100% of the display dimensions in an ortho perspective

Built in Shortcuts:

s - Take screenshot and save it in 'resources/screenshots' as a PNG
ESC / q - quit application / terminate window (if used with looper.py)

'''
import numpy as np

import pygame as pg
from pygame.font import *
from pygame import *
from pygame.constants import *

from OpenGL.GL.shaders import *
from OpenGL.GL import *
from OpenGL.GLU import *

from .opencv_window import Window
import cv2 as cv

class OpenGLWindow(Window):

    def __init__(self, width, height, window_name='image'):
        self.width = width
        self.height = height
        self.window_name = window_name
        self.last_image = None
        self.show_fps = False
        self.ui_color = (0, 255, 0)
        self.ui_text = {}
        self.ui_font = cv.FONT_HERSHEY_PLAIN
        self.shader = None
        self.verts = None

        self.window = pg.display.set_mode((width, height), DOUBLEBUF | OPENGL)
        pg.init()

        self.init_gl()
        self.init_texture()
        self.init_verts()
        self.init_shaders()



    def render_img(self, img):
        glClearColor(0.25, 0.25, 0.25, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.shader)
        glDrawArrays(GL_QUADS, 0, 4)

        # final = display_fps(clock, img)
        self.gl_draw_image(img)
        pg.display.flip()


    def gl_draw_image(self, img):

        # bind texture
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        # load texture
        glTexSubImage2D(
            GL_TEXTURE_2D,
            0,
            0, 0,
            img.shape[1],
            img.shape[0],
            GL_RGB,  # the format of the input
            GL_UNSIGNED_BYTE,
            img.data
        )

        # setting up texture parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        # set texture filtering mode
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

        # glBindTexture(GL_TEXTURE_2D, 0)


    def init_gl(self):

        gluPerspective(0, (self.width / self.height), 0.1, 50.0)
        glOrtho(-1, 1, 1, -1, -1, 4)
        glTranslatef(0.0, 0.0, -5)

        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)


    def init_texture(self):
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)

        img = np.zeros([self.height, self.width, 3], dtype=np.uint8)

        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGB,  # how the texture is stored in memory
            self.width,
            self.height,
            0,
            GL_RGB,  # the format of the input
            GL_UNSIGNED_BYTE,
            img.data
        )
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        # set texture filtering mode
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

        glBindTexture(GL_TEXTURE_2D, 0)

    def init_shaders(self):
        # Load shaders
        vertexShader = compileShader(self.get_file_content("gl_test_shader.vert"), GL_VERTEX_SHADER)
        fragmentShader = compileShader(self.get_file_content("gl_test_shader.frag"), GL_FRAGMENT_SHADER)

        shaderProgram = glCreateProgram()
        glAttachShader(shaderProgram, vertexShader)
        glAttachShader(shaderProgram, fragmentShader)
        glLinkProgram(shaderProgram)
        glUseProgram(shaderProgram)

        self.shader = shaderProgram

        return shaderProgram


    def init_verts(self):
        """We will draw a full screen Quad with a single texture, which will be used to display any
        image output on this window.
        """

        vertices = [-1, 1,
                    1, 1,
                    1, -1,
                    -1, -1]

        texcoords = [0.0, 0.0,
                     1.0, 0.0,
                     1.0, 1.0,
                     0.0, 1.0

                     ]

        vertices = np.array(vertices, dtype=np.float32)
        texcoords = np.array(texcoords, dtype=np.float32)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, vertices)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, texcoords)
        glEnableVertexAttribArray(0)
        glEnableVertexAttribArray(1)

    def get_file_content(self, file):
        content = open(file, 'r').read()
        return content

    def capture_key(self):
        for event in pg.event.get():  # process events since last loop cycle
            if event.type == pg.QUIT:
                pg.quit()
                quit()

            if event.type == KEYDOWN:
                print("Keydown")

                return event.key

        return False # no key was pressed
