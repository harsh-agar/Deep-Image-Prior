import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import cv2

from .common_utils import *

def get_text_mask(for_image, sz=20):
    font_fname = '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf'
    font_size = sz
    font = ImageFont.truetype(font_fname, font_size)

    img_mask = Image.fromarray(np.array(for_image)*0+255)
    draw = ImageDraw.Draw(img_mask)
    draw.text((128, 128), "hello world", font=font, fill='rgb(0, 0, 0)')

    return img_mask

def get_bernoulli_mask(for_image, zero_fraction=0.95):
    img_mask_np=(np.random.random_sample(size=pil_to_np(for_image).shape) > zero_fraction).astype(int)
    img_mask = np_to_pil(img_mask_np)
    
    return img_mask

def auto_canny(image, sigma=0.2):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

def get_edge_mask(for_image, num_grids=0, increase_contrast=False, dilation=0, edge_sigma=0.33):
    for_image = np.transpose(for_image, (1,2,0))
    for_image = (for_image.copy() * 255).astype('uint8')
    if increase_contrast:
        lab   = cv2.cvtColor(for_image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        cl    = clahe.apply(l_channel)
        limg  = cv2.merge((cl, a, b))
        for_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    img_grid = np.zeros(for_image.shape[:-1])
    if num_grids:
        img_grid[:, ::int(img_grid.shape[0]/num_grids)] = 255
        img_grid[::int(img_grid.shape[0]/num_grids), :] = 255

    gray = cv2.cvtColor(for_image, cv2.COLOR_BGR2GRAY)
    img_edges = auto_canny(gray, sigma=edge_sigma)
    
    final_edge = (((img_edges/255 + img_grid/255) > 0) * 255).astype('uint8')
        
    if dilation:
        final_edge = cv2.dilate(final_edge.copy(), None, iterations=dilation)
    return np.expand_dims(final_edge.astype('float32')/255, axis=0)

def generate_mask_by_percent(mask_shape, data_percentage=2):
    mask_img = np.zeros(mask_shape)
    valid_px = np.random.randint(mask_shape[1], size=(int(data_percentage * mask_shape[0] / 100), mask_shape[1]))
    for row_n in range(mask_shape[1]):
        mask_img[row_n, valid_px[:,row_n]] = 1.0
    return np.expand_dims(np.float32(mask_img), axis=0)

