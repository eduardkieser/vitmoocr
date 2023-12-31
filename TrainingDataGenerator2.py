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
from enum import Enum

class ColorScheme(Enum):
    random = 0
    light_back = 1
    dark_back = 2

class Config():
    def __init__(self):
        self.width = 900
        self.height = 900
        self.fps = 1
        self.fpb = 5
        self.seconds = int(24*60*60)
        self.color_scheme = ColorScheme.random
        self.adjacent_chars = ['(',')','/','|']
        self.n_frames_per_screen = 10
        self.font_size_range = (40, 120)
        self.number_range = (0, 200)
        self.include_number_noise = True
        self.number_noise_font_range=(3, 6)
        self.size_delta = (0,0)
        self.add_qr_corners=True
        self.data_path='/data'
        self.path_to_fonts='fonts'
        self.qr_box_size = 5
        self.qr_border = 4
        self.qr_size = self.qr_box_size * 21 + 2 * self.qr_box_size * self.qr_border
        self.add_qr = True

class Frame():
    def __init__(self, left, right, top, bottom, font_size=None, font_path=None):
        height = bottom - top
        width = right - left
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.width = width
        self.height = height
        self.top_left = (left, top)
        self.bottom_right = (right, bottom)
        self.size = (width, height)
        self.font_size = font_size
        self.font_path = font_path
        self.center = (int((left + right) / 2), int((top + bottom) / 2))


class ScreenGenerator():
    def __init__(self):
        self.config = Config()
        self.frames = []
        self.numbers = []
        self.selected_numbers = []
        self.font_paths = \
            glob(self.config.path_to_fonts + '/*.ttf') + \
            glob(self.config.path_to_fonts + '/*.otf') # + \
        
        self.center_frame = None
        self.foreground_color = None
        self.background_color = None
        self.refresh_colors()

    def get_text_size(self, font_size, font_path='vitmoocr/fonts/DejaVuSans-Bold.ttf'):
        font = ImageFont.truetype(font_path, font_size)
        img = Image.new('RGB', (300, 300))
        draw_txt = ImageDraw.Draw(img)
        width, height = draw_txt.textsize('250', font=font)
        return width, height

    def shuffle_fonts(self):
        for frame in self.frames:
            np.random.shuffle(self.font_paths)
            font_path = self.font_paths[0]
            frame.font_path = font_path

    def compose_screen(self):
        screen_w, screen_h = self.config.width, self.config.height
        qr_size = self.config.qr_size
        self.frames = []
        self.center_frame = Frame(
            left=int(screen_w / 2 - qr_size / 2),
            right=int(screen_w / 2 + qr_size / 2),
            top=int(screen_h / 2 - qr_size / 2),
            bottom=int(screen_h / 2 + qr_size / 2),
        )
        TL_frame = Frame(
            top=0,left=0,
            bottom=qr_size,right=qr_size
        )
        BR_frame = Frame(
            top=screen_h-qr_size,
            left=screen_w-qr_size,
            bottom=screen_h,
            right=screen_w
        )

        n_frames_per_screen = self.config.n_frames_per_screen
        n_failiors = 0

        while (len(self.frames) < n_frames_per_screen) & (n_failiors < n_frames_per_screen*5):
            overlaps = False

            # np.random.shuffle(self.font_paths)
            font_path = self.font_paths[0]
            font_size = np.random.randint(self.config.font_size_range[0], self.config.font_size_range[1])
            # Randomly choose a top left corner
            text_width, text_height = self.get_text_size(font_size, font_path)
            text_width, text_height = (text_width+self.config.size_delta[0]),(text_height+self.config.size_delta[1])
            top_candidate = np.random.randint(0, screen_h - text_height)
            left_candidate = np.random.randint(0, screen_w - text_width)

            center_x_candidate = int(left_candidate+text_width/2)
            center_y_candidate = int(top_candidate+text_height/2)
            # check if the candidate overlaps with any existing frames or qr
            for frame in self.frames + [self.center_frame, TL_frame,BR_frame]:
                overlap_vert = top_candidate in range(frame.top - text_height, frame.bottom)
                overlap_hors = left_candidate in range(frame.left - text_width, frame.right)

                x_delta = abs(frame.center[0]-center_x_candidate)
                y_delta = abs(frame.center[1]-center_y_candidate)

                if (x_delta<qr_size) and (y_delta<qr_size):
                    # qr boxes overlap
                    overlaps = True
                    break

                if overlap_vert and overlap_hors:
                    # numbere boxes overlap
                    n_failiors = n_failiors + 1
                    overlaps = True
                    break
            if not overlaps:
                self.frames.append(Frame(
                    left=left_candidate,
                    right=left_candidate + text_width,
                    top=top_candidate,
                    bottom=top_candidate + text_height,
                    font_size=font_size,
                    font_path=font_path
                ))

    def generate_random_nan(self):
        chars = string.ascii_letters
        size = np.random.randint(1, 4)
        return ''.join(choice(chars) for _ in range(size))

    def generate_image_id_set(self):
        chars = string.ascii_letters+string.digits
        size = 6
        image_id = ''.join(choice(chars) for _ in range(size))
        return 'nums_' + image_id, 'labs_' + image_id

    def generate_numbers(self, n=1000):
        n = n / 65
        numbers = []
        # X XX XXX nan
        weights = [3, 4, 5, 3]
        # X
        for i in range(int(n * weights[0] / np.sum(weights))):
            numbers = numbers + list(range(0, 10))
        # XX
        for i in range(int(n * weights[1] / np.sum(weights))):
            numbers = numbers + list(range(10, 100))
        # XXX
        for i in range(int(n * weights[2] / np.sum(weights))):
            numbers = numbers + list(range(100, 251))
            # nan
        for i in range(int(n * weights[2] / np.sum(weights))):
            numbers = numbers + [self.generate_random_nan()]

        shuffle(numbers)
        self.numbers = numbers

    def inspect_numbers(self):
        plot_numbers = [num if type(num) is int else -1 for num in self.numbers]
        nums_s = pd.Series(plot_numbers)
        nums_s.hist(bins=nums_s.unique().size)

    def get_qr_img(self, text, size=None):
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=self.config.qr_box_size,
            border=self.config.qr_border,
        )
        qr.add_data(text)
        qr.make()
        img = qr.make_image(fill_color="black", back_color="white")
        img = self.change_perspective(img)
        return img

    def change_perspective(self, img):

        direction = 'top'

        img = np.array(img).astype(np.uint8)*255
        skew_factor = 0.1
        width, height = img.shape[:2]
        scale_factor = min(width, height)
        if direction=='right':
            skew_matrix = np.array([
                [0, 0],
                [0, 1],
                [0, 1],
                [0, 0]
            ])
            y_stretch_factor = -0.1
        if direction=='left':
            skew_matrix = np.array([
                [0, 1],
                [0, 0],
                [0, 0],
                [0, 1]
            ])
            y_stretch_factor = -0.1

        if direction=='top':
            skew_matrix = np.array([
                [1, 0],
                [1, 0],
                [0, 0],
                [0, 0]
            ])
            y_stretch_factor = 0.1

        if direction=='bottom':
            skew_matrix = np.array([
                [0, 0],
                [0, 0],
                [1, 0],
                [1, 0]
            ])
            y_stretch_factor = 0.1


        y_stretch_matrix = np.array([
            [0, -1],
            [0, -1],
            [0, -1],
            [0, -1]
        ])

        a = skew_matrix * skew_factor * scale_factor + \
            y_stretch_matrix * y_stretch_factor * scale_factor

        pts1 = np.float32([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ])
        pts2 = np.float32([
            [0 + a[0, 0], 0 + a[0, 1]],
            [width - a[1, 0], 0 + a[1, 1]],
            [width - a[2, 0], height - a[2, 1]],
            [0 + a[3, 0], height - a[3, 1]]
        ])
        M = getPerspectiveTransform(pts1, pts2)
        img = warpPerspective(img, M, (height, width), borderValue=(255, 255, 255))
        return Image.fromarray(img)

    def paint_frame_outlines_and_codes(self, numbers, img=None):
        qr_size = self.config.qr_size
        if img is None:
            color = (117,117,117)
            screen_size = (self.config.width,self.config.height)
            img = img = Image.new(mode="RGB", size=screen_size, color = color)
        draw = ImageDraw.Draw(img)
        for ix, frame in enumerate(self.frames):
            number = numbers[ix]
            if type(number) is int:
                number = "{:>3d}".format(number)
            draw.rectangle((frame.top_left, frame.bottom_right), outline='white', width=2)
            data = f'{numbers[ix]}_{frame.width}_{frame.height}'
            code_img = self.get_qr_img(data)
            img.paste(code_img, box=(frame.center[0] - int(qr_size / 2), frame.center[1] - int(qr_size / 2)))
        return img

    def paint_frame_outlines(self, img=None):
        if img is None:
            img = img = Image.new(mode="RGB", size=self.screen_size)
        draw = ImageDraw.Draw(img)
        for ix, frame in enumerate(self.frames):
            draw.rectangle((frame.top_left, frame.bottom_right), outline='white', width=2)
        return img

    def paint_geometric_noise(self,img=None, n_geometries = 20):
        screen_size = self.config.width, self.config.height

        if img is None:
            img = Image.new("RGB", screen_size, color=self.background_color)

        line_color = self.get_background_ish_color()
        fill_color = self.get_background_ish_color()

        draw = ImageDraw.Draw(img)
        for geo_ix in range(n_geometries):
            center = (np.random.randint(0, screen_size[0]), np.random.randint(0, screen_size[1]))
            width = np.random.randint(0, screen_size[0] / 5)
            height = np.random.randint(0, screen_size[0] / 5)
            geometry = np.random.randint(0, 3)
            fill = bool(random.getrandbits(1))
            line_weight = np.random.randint(1,4)
            top_left = (center[0]-width/2,center[1]-height/2)
            bottom_right = (center[0] + width / 2, center[1] + height / 2)

            if geometry == 0:
                draw.rectangle((top_left, bottom_right), fill=fill_color, outline=line_color, width=line_weight)
            if geometry == 1:
                draw.ellipse((top_left, bottom_right), fill=fill_color, outline=line_color, width=line_weight)

        return img


    def paint_text_noise(self, img=None, n=20):
        screen_size = self.config.width, self.config.height
        if img is None:
            img = Image.new("RGB", screen_size, color=self.background_color)
        draw = ImageDraw.Draw(img)
        text_color = self.get_foreground_contrast_color()
        font_path = self.font_paths[0]
        font_size = int(np.random.randint(self.config.font_size_range[0], self.config.font_size_range[1])/3)
        font = ImageFont.truetype(font_path, font_size)
        for ix in range(n):
            top_left = (np.random.randint(0, screen_size[0]), np.random.randint(0, screen_size[1]))
            text = self.generate_random_nan()
            draw.text(top_left, str(text), font=font, fill=self.foreground_color)

        return img


    def paint_id_and_corner_markers(self, frame_id, img=None):
        width, height = self.config.width, self.config.height
        screen_size = width, height
        qr_size = self.config.qr_size
        if img is None:
            img = Image.new(mode="RGB", size=screen_size)
        code_img = self.get_qr_img('TL')
        img.paste(code_img, box=(0, 0))
        code_img = self.get_qr_img('BR')
        img.paste(code_img, box=(width - qr_size, height - qr_size))
        code_img = self.get_qr_img(frame_id)
        img.paste(code_img, box=(int((width - qr_size) / 2), int((height - qr_size) / 2)))
        return img

    def paint_numbers_to_frames(self, numbers, img=None):
        screen_size = self.config.width, self.config.height
        if img is None:
            img = Image.new("RGB", screen_size, color=self.background_color)
        draw = ImageDraw.Draw(img)
        for ix, frame in enumerate(self.frames):
            number = numbers[ix]
            if type(number) is int:
                number = "{:>3d}".format(number)
            font = ImageFont.truetype(frame.font_path, frame.font_size)
            top_left = (frame.top_left[0]+self.config.size_delta[0]/2, frame.top_left[1]+self.config.size_delta[1]/2)
            number_str = self.add_adjacent_charachters(str(number))
            draw.text(top_left, number_str, font=font, fill=self.foreground_color)
        return img

    def add_adjacent_charachters(self, number_str):
        chars = self.config.adjacent_chars
        shuffle(chars)
        if random.randint(0,10)<3:
            number_str = chars[0]+number_str
        if random.randint(0,10)<3:
            number_str = number_str + chars[1]
        return number_str
        

    def refresh_colors(self):
        def get_dark_color():
            return (random.randint(0, 117),
                    random.randint(0, 117),
                    random.randint(0, 117))

        def get_light_color():
            return (random.randint(117, 255),
                    random.randint(117, 255),
                    random.randint(117, 255))

        if bool(random.getrandbits(1)):
            self.foreground_color = get_light_color()
            self.background_color = get_dark_color()
        else:
            self.foreground_color = get_dark_color()
            self.background_color = get_light_color()

    def get_random_color(self):
        return (random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255))

    def get_foreground_contrast_color(self):
        f1,f2,f3 = self.foreground_color
        return (
            (f1+random.randint(40, 150))%255,
            (f2+random.randint(40, 150))%255,
            (f3+random.randint(40, 150))%255)

    def get_background_ish_color(self):
        f1, f2, f3 = self.background_color
        rg = 50
        return(
            np.clip(f1 + random.randint(-rg, rg), 0, 255),
            np.clip(f2 + random.randint(-rg, rg), 0, 255),
            np.clip(f3 + random.randint(-rg, rg), 0, 255),
        )

    def sample_numbers(self):
        if self.numbers == []:
            self.generate_numbers()
        self.selected_numbers = sample(self.numbers, self.config.n_frames_per_screen)

    def generate_numbers_img(self, numbers_id):

        numbers_img = self.paint_geometric_noise()
        numbers_img = self.paint_text_noise(numbers_img)
        numbers_img = self.paint_numbers_to_frames(self.selected_numbers,numbers_img)
        numbers_img = self.paint_id_and_corner_markers(numbers_id, numbers_img)
        # numbers_img = self.paint_frame_outlines(numbers_img)
        return numbers_img

    def generate_labels_img(self, labels_id):
        if self.selected_numbers == []:
            self.sample_numbers()
        labels_img = self.paint_frame_outlines_and_codes(self.selected_numbers)
        labels_img = self.paint_id_and_corner_markers(labels_id, labels_img)
        return labels_img

    def generate_image_set(self):
        if self.selected_numbers == []:
            self.sample_numbers()
        numbers_id, labels_id = self.generate_image_id_set()
        numbers_img = self.generate_numbers_img(numbers_id)
        labels_img = self.generate_labels_img(labels_id)
        return numbers_img, labels_img



class VitmoVideoWriter():
    def __init__(
        self,
        config = Config()
    ):
        
        self.width = config.width
        self.height = config.height
        self.fps = config.fps
        self.fpb = config.fpb
        self.seconds = config.seconds

    def generate_video(self):
                fourcc = VideoWriter_fourcc(*'MP42')
                video = VideoWriter('./take2.avi', fourcc, float(self.fps), (self.width, self.height))
                generator = ScreenGenerator(
                    n_frames_per_screen=5,
                    screen_size=(self.width, self.height),
                )
                generator.compose_screen()

                for i in range(self.fps * self.seconds):
                    print(i)
                    generator.sample_numbers()
                    generator.shuffle_fonts()
                    generator.refresh_colors()
                    numbers_img, labels_img = generator.generate_image_set()
                    video.write(np.asarray(labels_img))
                    video.write(np.asarray(numbers_img))
                    for j in range(self.fpb):
                        generator.shuffle_fonts()
                        generator.refresh_colors()
                        numbers_img, labels_img = generator.generate_image_set()
                        video.write(np.asarray(numbers_img))
                video.release()

    def play_video(self):
        frame_name = 'look here'
        # imshow(frame_name, np.zeros((1000,1000,3)))
        # waitKey(0)
        isFirstFrame = True
        video_frame_generator = self.video_frame_generator()
        period_in_ms = int((1/self.fps)*1000)
        start_time = time.time()
        for frame in video_frame_generator:
            if isFirstFrame:
                imshow(frame_name, frame)
                waitKey(0)
                isFirstFrame=False
            imshow(frame_name, frame)
            waitKey(period_in_ms)

        n_batches = int((self.fps * self.seconds) / self.fpb)
        n_frames = n_batches*self.fpb
        time_to_play = time.time()-start_time
        print(f'generated {n_frames} frames in {time_to_play} seconds')
        print(f'mean_period: {time_to_play/n_frames}')
        print(f'mean_fps: {n_frames/time_to_play} while aiming for {self.fps}')

    def video_frame_generator(self):
        n_batches = int((self.fps*self.seconds)/self.fpb)
        frame_generator = ScreenGenerator()

        for i in range(n_batches):
            print(f'{int(((n_batches-i)/n_batches)*100)}% left')
            is_first_frame = 0
            frame_generator.sample_numbers()
            frame_generator.shuffle_fonts()
            frame_generator.refresh_colors()
            frame_generator.compose_screen()
            numbers_id, labels_id = frame_generator.generate_image_id_set()
            labels_img = frame_generator.generate_labels_img(labels_id)
            while is_first_frame<1:
                is_first_frame = is_first_frame+1
                yield np.asarray(labels_img)

            for j in range(self.fpb):
                frame_generator.shuffle_fonts()
                frame_generator.refresh_colors( )
                numbers_img = frame_generator.generate_numbers_img(numbers_id)
                yield np.asarray(numbers_img)


if __name__=='__main__':
    # generator = ScreenGenerator(n_frames_per_screen=20)
    # generator.compose_screen()
    # generator.refresh_colors()
    # numbers_img, labels_img = generator.generate_image_set()
    # sbs_img = Image.new(mode="RGB",size=(generator.screen_size[0]*2,generator.screen_size[1]))
    # sbs_img.paste(numbers_img)
    # sbs_img.paste(labels_img, box=(generator.screen_size[0],0))
    # draw = ImageDraw.Draw(sbs_img)
    # draw.rectangle(((0,0), generator.screen_size), outline='white', width=2)
    # sbs_img

    VitmoVideoWriter().play_video()