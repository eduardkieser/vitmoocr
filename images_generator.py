

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
import numpy as np
# import matplotlib.pyplot as plt
import random
from glob import glob
import os
import shutil



def create_noise_mask(width,height):
    mask = np.random.uniform(-10,10,(height,width,3)).astype(np.uint8)
    return mask


def get_gaussian_mask(width,height):

    center = (random.randint(-10,width+10),random.randint(-10,width+10))

    dia = random.randint(1,width)

    image = Image.new('RGB', (width, height), (0,0,0) )
    draw = ImageDraw.Draw(image)
    draw.ellipse((center[0]-dia, center[1]-dia, center[0]+180, center[1]+180),\
                 fill=(5,5,5), outline='white')

    image = image.filter(ImageFilter.GaussianBlur(radius=random.randint(15,40)))

    image = (np.array(image)).astype(np.uint8)

    # plt.imshow(Image.fromarray(image))

    return image

def generate_image(font, text, rotation, shape,text_color,back_color,text_offset, final_shape=(28,28)):

    im_width = shape[0]
    im_height = shape[1]

    txt_img = Image.new('L', (im_width, im_height))
    bak_img = Image.new('RGB', (im_width, im_height), color=back_color)

    d = ImageDraw.Draw(txt_img)
    text_width, text_height = d.textsize(text,font)
    corner_coords = (int(im_width/2-text_width/2), int(im_height/2-text_height/2)-10)

    d.text(corner_coords, text, fill=(255),font=font)

    txt_img = np.array(txt_img)

    border_width = np.clip(np.random.randint(-30,20,4), 0, 30)
    txt_img[:,0:border_width[0]] = 255
    txt_img[:,-(border_width[1]+1):] = 255
    txt_img[0:border_width[0],:] = 255
    txt_img[-(border_width[1]+1):,:] = 255


    txt_img = Image.fromarray(txt_img)
    txt_img = txt_img.rotate(rotation)
    bak_img.paste(ImageOps.colorize(txt_img, text_color, text_color), text_offset,  txt_img)
    img = bak_img
    img = img.filter(ImageFilter.BLUR)
    img = img.resize((im_width, im_height))
    # noise_mask = create_noise_mask(im_width, im_height)
    gaussian_mask = get_gaussian_mask(im_width, im_height)
    np_img = np.array(img)
    # np_img = np_img+noise_mask
    np_img = np_img+gaussian_mask
    np_img = np.clip(np_img,0,255)
    im_mean = np_img.mean()
    im_std = np_img.std()
    img = Image.fromarray(np_img,'RGB')
    img = img.resize(final_shape)

    return img

def get_dark_color():
    return (random.randint(0,117),
            random.randint(0,117),
            random.randint(0,117))

def get_light_color():
    return (random.randint(117,255),
            random.randint(117,255),
            random.randint(117,255))


def create_image_batch(i_stop, folder_path, final_size, start_fresh):

    i_start = 0;

    if start_fresh:
        if os.path.isdir(folder_path):
            shutil.rmtree(folder_path)

    font_paths =glob('./vitmoocr/fonts/*.ttf')

    final_shape = (final_size,final_size)
    
    for i in range(i_start,i_stop):

        rotation = random.randint(-20,20)
        text_offset = (random.randint(-30,30),random.randint(-30,30))
        number = -1
        while number<0:
            number = int(np.random.triangular(-20,150,200))
        font_size = random.randint(80,110)
        font_ix = random.randint(0,len(font_paths)-1)
        font = ImageFont.truetype(font_paths[font_ix],font_size)

        if i%2==0:
            text_color = get_dark_color()
            back_color = get_light_color()
        else:
            text_color = get_light_color()
            back_color = get_dark_color()

        original_shape = (random.randint(200,300),random.randint(150,200))


        img = generate_image(font=font,
                             text=str(number),
                             rotation=rotation,
                             shape=original_shape,
                             back_color=back_color,
                             text_color=text_color,
                             text_offset=text_offset,
                             final_shape=final_shape
                             )

        """
        folder_path: 'data/training_data' -> number_path: '..data/training_data/178'
        """
        number_path = os.path.join(folder_path, str(number))

        if not(os.path.isdir(number_path)):
            os.makedirs(number_path)

        img.save(os.path.join( number_path,f"{number}-{i}.jpg") )

        if i%100==0:
            print(i)


def create_data_batch(image_size=96, train_batch_size = 1000, test_batch_size = 100):

    final_size = 96

    folder_path = os.path.join('data','training_data');
    create_image_batch(i_stop=train_batch_size, folder_path=folder_path, final_size=final_size,start_fresh=True)

    folder_path = os.path.join('data','testing_data');
    create_image_batch(i_stop=test_batch_size, folder_path=folder_path, final_size=final_size,start_fresh=True)

if __name__=='__main__':

    create_data_batch();
    # folder_name = 'testing_data'
    # create_image_batch(0,100, folder_name=folder_name, final_size=final_size)
