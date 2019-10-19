from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps, ImageColor
from glob import glob
import string
from random import choice, shuffle, sample
import numpy as np
import pandas as pd
import qrcode
import random
from cv2 import VideoWriter, VideoWriter_fourcc, imshow, waitKey, getPerspectiveTransform, warpPerspective
import time
from scipy import signal
from typing import List, Tuple



class Frame():
    def __init__(self, center, screen_size,text='0', font_size=20, font_path=None,text_color=(255,255,255)):
        self.text = text
        self.center:List[float] = center
        self.text_color = text_color
        self.screen_size = screen_size
        self.font = ImageFont.truetype(font_path, font_size)
        self.size = self.get_bounding_box_size()
        self.top_left = (self.center[0]*screen_size[0]-self.size[0]/2),(self.center[1]*screen_size[1]-self.size[1]/2)

    def get_bounding_box_size(self)->Tuple[int,int]:
        font = self.font
        img = Image.new('RGB', (300, 300))
        draw_txt = ImageDraw.Draw(img)
        width, height = draw_txt.textsize('250', font=font)
        return width, height


class Config():
    def __init__(self):
        font_size = 40
        self.background_color = (200, 200, 200)
        font_path = 'fonts/DejaVuSans-Bold.ttf'
        self.screen_size = 800, 800
        text_color = (0,0,0)
        self.frames :List[Frame] = [
            Frame(center=[0.25, 0.25], font_size=90,font_path=font_path, screen_size=self.screen_size,
                  text_color=text_color),
            Frame(center=[0.25, 0.75], font_size=80,font_path=font_path, screen_size=self.screen_size,
                  text_color=text_color),
            Frame(center=[0.75, 0.25], font_size=80,font_path=font_path, screen_size=self.screen_size,
                  text_color=text_color),
            Frame(center=[0.75, 0.75], font_size=120,font_path=font_path, screen_size=self.screen_size,
                  text_color=text_color)
        ]

        t = np.linspace(0, 1, 60*10)
        saw = (signal.sawtooth(2*np.pi*5*t)+1)* (250/2)
        square = (signal.square(2*np.pi*5*t)+1)* (250/2)
        sin = (np.sin(2*np.pi*5*t)+1)* (250/2)
        random = np.random.randint(0,250,60*10)
        self.signals : List[np.array] = [saw,sin,sin,saw]

class ICUReplica:
    def __init__(self,config: Config):


        self.config = config

    def create_screen(self,ix):
        img = Image.new(mode="RGB", size=self.config.screen_size, color=self.config.background_color)
        draw = ImageDraw.Draw(img)
        for (frame, sig) in zip(self.config.frames,self.config.signals):
            draw.text(
                (frame.top_left[0],frame.top_left[1]),
                str(int(sig[ix])),
                font=frame.font,
                fill=frame.text_color)
        return img


    def frame_generator(self):
        for ix in range(self.config.signals[0].shape[0]):
            img = self.create_screen(ix)
            yield np.asarray(img)


    def play_video(self, seconds = 1800):
        frame_name = 'ICU replica v0.0.1'
        isFirstFrame = True
        frame_generator = self.frame_generator()
        period_in_ms = int((1/1)*1000)
        for frame in frame_generator:
            if isFirstFrame:
                imshow(frame_name, frame)
                waitKey(0)
                isFirstFrame=False
            imshow(frame_name, frame)
            waitKey(period_in_ms)


if __name__=='__main__':

    ICUReplica(Config()).play_video(60*60*24)