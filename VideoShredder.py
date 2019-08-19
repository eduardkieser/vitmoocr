from pyzbar.pyzbar import decode as decodeqr
from vitmoocr.TrainingDataGenerator2 import ScreenGenerator
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
from glob import glob
import string
from random import choice

class LabeledRoi():
    def __init__(self, data, center, size):
        self.data = data
        self.center = center
        self.width = size[0]
        self.height = size[1]
        self.top = center[1] - size[1] / 2
        self.bottom = center[1] + size[1] / 2
        self.left = center[0] - size[0] / 2
        self.right = center[0] + size[0] / 2


class LabelScreen():
    def __init__(self, screen_id):
        self.screen_id = screen_id
        self.labeled_rois = []
        self.marker_tl = None
        self.marker_br = None

    def get_corner_coords(self):
        if self.marker_br is None or self.marker_tl is None:
            return None
        return (self.marker_tl.center, self.marker_br.center)


class VideoShredder():

    def __init__(self, file_path):
        self.output_dir = '/data2/'
        self.videos_dir = '/Users/eduard/workspaces/ml_projects/keras/VitmoOCR/videos/'
        self.decoded_ids = []
        self.last_valid_label_screen = None
        self.file_path = file_path
        self.rotate_angle = 90
        self.codes_read = 0
        # self.size_delta = (-25,-25)
        self.size_delta = (0, 0)
        self.show_snipping = True
        self.show_th_frames = True
        self.marker_tl = None
        self.marker_br = None
        self.xi=None
        self.yi=None


    def get_corner_coords(self):
        if self.marker_br is None or self.marker_tl is None:
            return None
        return (self.marker_tl.center, self.marker_br.center)

    def open_video_batch_and_start_shredding(self):
        file_list = glob(self.videos_dir+'*.mp4')
        for video in [
            # '/Users/eduard/workspaces/ml_projects/keras/VitmoOCR/videos/Redmi Note 4/VID_20190815_191730.mp4',
            #'/Users/eduard/workspaces/ml_projects/keras/VitmoOCR/videos/Redmi Note 4/VID_20190815_224328.mp4',
            #'/Users/eduard/workspaces/ml_projects/keras/VitmoOCR/videos/Redmi Note 4/VID_20190815_231255.mp4'
            # '/Users/eduard/workspaces/ml_projects/keras/VitmoOCR/videos/Moto/done/VID_20190815_182923.mp4',
            # '/Users/eduard/workspaces/ml_projects/keras/VitmoOCR/videos/Moto/done/VID_20190815_191730.mp4',
            # '/Users/eduard/workspaces/ml_projects/keras/VitmoOCR/videos/Moto/done/VID_20190815_224329.mp4'

            # not shure about these
            # '/Users/eduard/workspaces/ml_projects/keras/VitmoOCR/videos/untitled folder/VID_20190807_184355.mp4',
            # '/Users/eduard/workspaces/ml_projects/keras/VitmoOCR/videos/untitled folder/VID_20190807_192311.mp4',
            # '/Users/eduard/workspaces/ml_projects/keras/VitmoOCR/videos/untitled folder/VID_20190807_192922.mp4',
            # '/Users/eduard/workspaces/ml_projects/keras/VitmoOCR/videos/untitled folder/VID_20190807_192956.mp4',
            # '/Users/eduard/workspaces/ml_projects/keras/VitmoOCR/videos/untitled folder/VID_20190807_193827.mp4',
            # '/Users/eduard/workspaces/ml_projects/keras/VitmoOCR/videos/untitled folder/VID_20190810_125157.mp4',


            # '/Users/eduard/workspaces/ml_projects/keras/VitmoOCR/videos/untitled folder/VID_20190811_183621_sms.mp4',
            # '/Users/eduard/workspaces/ml_projects/keras/VitmoOCR/videos/untitled folder/VID_20190811_183622_rn4.mp4',
            '/Users/eduard/workspaces/ml_projects/keras/VitmoOCR/videos/untitled folder/VID_20190811_183626_mto.mp4',



            ]:
            print(f'decoding {video}')
            self.open_video_and_start_shredding(file_path=video)

    def get_corners(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f'left top yo, tl at {x},{y}')
            self.marker_tl = LabeledRoi('TL',(x,y),(50,50))
        if event == cv2.EVENT_RBUTTONDOWN:
            print(f'right buttom yo, br at {x},{y}')
            self.marker_br = LabeledRoi('BR',(x,y),(50,50))

    def open_video_and_start_shredding(self, file_path = None):
        cv2.namedWindow('frame')
        cv2.setMouseCallback('frame', self.get_corners)
        if file_path is None:
            file_path = self.file_path
        cap = cv2.VideoCapture(file_path)
        is_first_frame = True
        while (cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            if self.rotate_angle!=0:
                frame = self.rotate_image(frame)
            if is_first_frame:
                is_first_frame=False
                cv2.imshow('frame',frame)
                cv2.waitKey(0)
                # cv2.destroyAllWindows()
            try:
                self.next_screen(frame)
            except Exception as e:
                print(e)
                pass
        cap.release()
        print(f'codes read = {self.codes_read}')

    def rotate_image(self,img):

        if self.rotate_angle==90:
            out = cv2.transpose(img)
            out = cv2.flip(out, flipCode=1)
            return out

    def decode_label_frame(self, img, screen_id):
        # if screen_id == self.last_valid_label_screen.screen_id:
        #     return
        ls = LabelScreen(screen_id)
        decodedqrs = self.decode_qrs(img)
        for decodedqr in decodedqrs:
            data = decodedqr.data.decode("utf-8")
            rect = decodedqr.rect
            center = (rect.left + rect.width / 2, rect.top + rect.height / 2)
            tl = (rect.left, rect.top)
            br = (rect.left + rect.width), (rect.top + rect.height)
            qr_size = (rect.width, rect.height)

            if data == 'TL':
                ls.marker_tl = LabeledRoi(data, tl, qr_size)
                continue
            if data == 'BR':
                ls.marker_br = LabeledRoi(data, br, qr_size)
                continue
            if data[:4] == 'labs' or data[:4] == 'nums':
                continue

            frame_size = tuple(map(int, data.split('_')[1:3]))

            ls.labeled_rois.append(LabeledRoi(
                data=data,
                center=center,
                size=frame_size
            ))
        if ls.get_corner_coords() is None:
            print('could not find frame corners')
            if self.get_corner_coords() is None:
                print('pre_set_corners not found')
                self.last_valid_label_screen = None
            else:
                print(f'compromize si: {ls.screen_id}')
                ls.marker_tl = self.marker_tl
                ls.marker_br = self.marker_br
                self.last_valid_label_screen = ls
        else:
            self.last_valid_label_screen = ls
            tl,br = ls.get_corner_coords()
            self.marker_tl, self.marker_br = LabeledRoi('TL',tl,(50,50)), LabeledRoi('BR',br,(50,50))

    def generate_random_id(self):
        chars = string.ascii_letters
        size = 6
        return ''.join(choice(chars) for _ in range(size))

    def chop_up_numbers_frame(self, img: int, screen_id):
        # this should be updated to take place at the point where the label frame is being added.
        # then we just check if the label fram is none
        if self.last_valid_label_screen is None:
            print('No valid label frame is available')
            return
        if screen_id != self.last_valid_label_screen.screen_id:
            print(f'screen_id: {screen_id}')
            print(f'self.last_valid_label_screen.screen_id: {self.last_valid_label_screen.screen_id}')
            print('Id does not match the last valid label frame, bouncing!')
            return
        print(f'Success, snipping {screen_id}')
        img = Image.fromarray(img)
        ws = 1#img.width / (
               #     self.last_valid_label_screen.marker_br.center[0] - self.last_valid_label_screen.marker_tl.center[0])
        hs = 1#img.height / (
               #     self.last_valid_label_screen.marker_br.center[1] - self.last_valid_label_screen.marker_tl.center[1])
        # print(len(self.last_valid_label_screen.labeled_rois))
        for roi in self.last_valid_label_screen.labeled_rois:
            frame = img.crop((roi.left-self.size_delta[0], roi.top-self.size_delta[1], roi.right+self.size_delta[0], roi.bottom+self.size_delta[1]))
            print('crop sucsess')
            if self.show_snipping:
                cv2.imshow('mini', np.array(frame))
                cv2.waitKey(10)

            roi_number, roi_width, roi_height = roi.data.split('_')
            if not roi_number.isdigit():
                roi_number = 'nan'
            folder_path = os.path.join('../data3', roi_number)
            random_id = self.generate_random_id()
            file_name = f'{roi_width}_{roi_height}_{screen_id}_{random_id}.png'
            if not (os.path.isdir(folder_path)):
                os.makedirs(folder_path)
            try:
                frame.save(os.path.join(folder_path, file_name))
            except Exception as e:
                print('we proabbly have a bad file name, trying again')
                file_name = file_name.replace("/","").replace(".","").replace("\\","")
                frame.save(os.path.join(folder_path, file_name)+'.png')

    def normalize_image(self, img):
        kernel_size = (100,100)
        step_size = (50,50)
        im_out = np.zeros(img.shape)
        width, height, depth = img.shape
        for layer in range(depth):
            for column_ix in range(int(width/kernel_size[0])+1):
                for row_ix in range(int(height/kernel_size[1])+1):
                    x0 = column_ix*kernel_size[0]
                    x1 = (column_ix+1)*kernel_size[0]
                    y0 = row_ix * kernel_size[1]
                    y1 = (row_ix + 1) * kernel_size[1]
                    # x1 = np.min([x1, width])
                    # y1 = np.min([y1, height])
                    window = img[x0:x1,y0:y1,layer]
                    window = cv2.normalize(window, None,alpha=0.0, beta=255.0, norm_type=cv2.NORM_MINMAX)
                    im_out[column_ix*kernel_size[0]:(column_ix+1)*kernel_size[0], row_ix * kernel_size[1]:(row_ix + 1) * kernel_size[1], 2-layer] = window

        im_out = (im_out).astype(np.uint8)
        # cv2.imshow('im_out',im_out)
        # cv2.waitKey(0)
        return im_out

    def threshold_image(self,img,th):
        if th ==-1:
            return np.array(img)
        mask = cv2.inRange(img, (0, 0, 0), (th, th, th))
        thresholded = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        img = 255 - thresholded
        return img

    def decode_qrs(self,img):
        # return decodeqr(img)
        results = []
        keys = []
        for theshold in list(range(130,231,20))+[-1]:
            th_img = self.threshold_image(img,theshold)
            if self.show_th_frames:
                cv2.imshow('frame',th_img)
                cv2.waitKey(50)
            new_results = decodeqr(th_img)
            for new_result in new_results:
                if new_result.data not in keys:
                    keys.append(new_result.data)
                    results.append(new_result)
        return results


    def read_screen_id(self, img):
        if img is None:
            return None
        results = self.decode_qrs(img)
        self.codes_read = self.codes_read+len(results)
        for result in results:
            code = result.data.decode("utf-8")
            if code[:4] == 'nums' or code[:4] == 'labs':
                return code
        return None

    def next_screen(self, img):
        screen_id = self.read_screen_id(img)
        print(f'screen id: {screen_id}')
        if screen_id is None:
            return None
        if screen_id[:4] == 'labs':
            self.decode_label_frame(img, screen_id[5:])
        if screen_id[:4] == 'nums':
            self.chop_up_numbers_frame(img, screen_id[5:])

if __name__=='__main__':
    # generator = ScreenGenerator(n_frames_per_screen=20)
    # generator.compose_screen()
    # numbers_img, labels_img = generator.generate_image_set()
    #
    decoder = VideoShredder(file_path='/Users/eduard/Desktop/VID_20190812_202349.mp4')
    decoder.open_video_batch_and_start_shredding()
    #     # decoder.next_screen(labels_img)
    #     # decoder.next_screen(numbers_img)