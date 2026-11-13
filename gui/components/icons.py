"""Small vector-style icons rendered locally for both appearance modes."""

import customtkinter as ctk
from PIL import Image, ImageDraw

from gui import theme as T


def sidebar_icon():
    def draw(color):
        image = Image.new("RGBA", (96, 96))
        pen = ImageDraw.Draw(image)
        pen.rounded_rectangle((14, 14, 82, 82), radius=12, outline=color, width=6)
        pen.line((38, 16, 38, 80), fill=color, width=6)
        pen.line((23, 31, 29, 31), fill=color, width=4)
        pen.line((23, 45, 29, 45), fill=color, width=4)
        return image

    return ctk.CTkImage(light_image=draw(T.TEXT[0]), dark_image=draw(T.TEXT[1]), size=(14, 14))
