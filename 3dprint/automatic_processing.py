from pathlib import Path
from tqdm import tqdm

import cv2
import numpy as np
import pymorph
# import skimage

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

def resize_by_larger_dim(img, w_ref=1600, h_ref=1400, display=False):

    h, w = img.shape

    if w >= h:
        width = w_ref
        height = None
        interp = cv2.INTER_AREA if w > w_ref else cv2.INTER_CUBIC  # AREA for shrinking, CUBIC for enlarging
    else:
        width = None
        height = h_ref
        interp = cv2.INTER_AREA if h > h_ref else cv2.INTER_CUBIC

    img_resized = resize(img, width, height)

    if display:
        cv2.imshow('Input', img)
        cv2.imshow('Resized', img_resized)
        cv2.waitKey(0)

    return img_resized

def threshold_img(img, display=False):

    th, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # ensure binary
    if display:
        cv2.imshow('threshold img', img)
        cv2.waitKey(0)

    return img

def inverse_img(img, display=False):

    inverse = 255 - img

    if display:
        cv2.imshow('inverse', inverse)
        cv2.waitKey(0)

    return inverse


def transparent_background(img, display=False):

    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    alpha = np.sum(bgr, axis=-1) > 0
    alpha = np.uint8(alpha * 255)
    result = np.dstack([bgr, alpha])  # Add the mask as alpha channel

    if display:
        cv2.imshow('transparent', result)
        cv2.waitKey(0)

    return result

def morph_thinning(img, iterations=3, se_size=6, display=False):
    """
    Code is based on:
    https://theailearner.com/tag/thickening-opencv-python/
    """
    # se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))

    thin = np.zeros(img.shape,dtype='uint8')
    img_out = img.copy()

    for n in range(iterations):
        # Erosion
        erode = cv2.erode(img_out ,se)
        # Opening on eroded image
        opening = cv2.morphologyEx(erode,cv2.MORPH_OPEN,se)
        # Subtract these two
        subset = erode - opening
        # Union of all previous sets
        thin = cv2.bitwise_or(subset,thin)
        # Set the eroded image for next iteration
        img_out = erode.copy()

    if display:
        cv2.imshow('original', img)
        cv2.imshow('thinned', thin)
        cv2.waitKey(0)

    return thin

def thicken_img(img, display=False):

    # img_binary = img.astype(bool)
    img_binary = pymorph.asbinary(img)
    img_out_binary = pymorph.thick(img_binary, Iab=None, n=1, theta=45, direction="clockwise")
    img_out = np.asarray(img_out_binary, dtype=np.uint8) * 255

    if display:
        cv2.imshow('original', img)
        cv2.imshow('thickened', img_out)
        cv2.waitKey(0)

    return img_out


def close_img(img, se_size=5, display=False):

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))
    # se = skimage.morphology.ball(se_size)
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se)

    if display:
        cv2.imshow('original', img)
        cv2.imshow('closed', closed)
        cv2.waitKey(0)

    return closed


def dilate_img(img, se_size=5, display=False):

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))
    closed = cv2.morphologyEx(img, cv2.MORPH_DILATE  , se)

    if display:
        cv2.imshow('original', img)
        cv2.imshow('dilated', closed)
        cv2.waitKey(0)

    return closed

def process_img(img_file, thick_type='dilate_closing', se_size=5, se_size_closing=15, display=0):

    img = cv2.imread(img_file, 0)

    img = resize_by_larger_dim(img, w_ref=1133, h_ref=1133, display=display>3)

    img = threshold_img(img, display=display>2)

    img = inverse_img(img, display=display>2)

    assert thick_type in ['closing', 'thickening', 'dilate', 'dilate_closing', 'closing_dilate']
    if thick_type == 'closing':
        img = close_img(img, se_size, display=display>1)
        name_sfx = '_{}_se_size_{:02d}'.format(thick_type, se_size)
    elif thick_type == 'thickening':
        img = thicken_img(img, display=display)
        name_sfx = '_{}_se_size_{:02d}'.format(thick_type, se_size)
    elif thick_type == 'dilate':
        img = dilate_img(img, se_size, display=display>1)
        name_sfx = '_{}_se_size_{:02d}'.format(thick_type, se_size)
    if thick_type == 'dilate_closing':
        img = close_img(img, se_size, display=display>1)
        img = dilate_img(img, se_size=se_size_closing, display=display>1)
        name_sfx = '_{}_se_size_{:02d}_{:02d}'.format(thick_type, se_size, se_size_closing)
    if thick_type == 'closing_dilate':
        img = dilate_img(img, se_size=se_size_closing, display=display>1)
        img = close_img(img, se_size, display=display>1)
        name_sfx = '_{}_se_size_{:02d}_{:02d}'.format(thick_type, se_size_closing, se_size)

    img = inverse_img(img, display=display>1)

    img = transparent_background(img, display=display>0)

    # save output image
    img_out_path = Path(img_file).parent.parent / 'output{}'.format(name_sfx) / Path(img_file).name
    img_out_path.parent.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(img_out_path.with_suffix('.png').as_posix(), img)

    if display > 0:
        cv2.destroyAllWindows()

    return img, img_out_path


def main():

    # img_ref_file = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/automatic_processing/reference.jpg'
    # img_ref = cv2.imread(img_ref_file, 0)
    # h_ref, w_ref = img_ref.shape

    img_file = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/automatic_processing/chicken.jpg'
    # img_file = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/automatic_processing/bear.jpg'
    # img_file = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/automatic_processing/ddfd.jpg'
    # img_file = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/automatic_processing/FFS.jpg'

    # display = False
    display = 0
    se_size = 5
    thick_type = 'closing'
    thick_type = 'thickening'

    process_img(img_file, thick_type, se_size, display)

    pass


def main_img_list():

    # img_ref_file = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/automatic_processing/reference.jpg'
    # img_ref = cv2.imread(img_ref_file, 0)
    # h_ref, w_ref = img_ref.shape

    input_dir = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/automatic_processing'

    img_name_list = ['chicken.jpg',
                     'bear.jpg',
                     'ddfd.jpg',
                     'FFS.jpg',
                     'magen_david_designed.jpg',
                     'reference.jpg',
                     'Spider_Laser cut.jpg',
                     'Untitled77755.jpg',
                     ]

    img_file = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/automatic_processing/ddfd.jpg'
    # img_file = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/automatic_processing/FFS.jpg'

    # display = False
    display = 0
    se_size = 10
    # thick_type = 'closing'
    thick_type = 'thickening'
    thick_type = 'dilate'


    for img_name in img_name_list:
        img_file = (Path(input_dir) / img_name).as_posix()
        process_img(img_file, thick_type, se_size, display)

    pass



def main_root_dir():

    # img_ref_file = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/automatic_processing/reference.jpg'
    # img_ref = cv2.imread(img_ref_file, 0)
    # h_ref, w_ref = img_ref.shape

    # input_dir = 'C:/Users/shay/Desktop/projects/envs/images/input'
    # input_dir = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/tests/input/'
    # input_dir = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/selected pic for testing -20230310T124257Z-001/selected pic for testing/'
    input_dir = 'C:/Users/shay/Desktop/KAVIM22.02.23/abstract.geometric'

    img_path_list = list(Path(input_dir).glob('*'))

    display = False
    # display = 5

    # thick_type_list = ['closing', 'dilate', 'dilate_closing', 'closing_dilate', 'thickening']
    # se_size_list = [3, 5, 10, 15]
    # se_size_closing = 5

    thick_type_list = ['closing', 'dilate', 'dilate_closing', 'closing_dilate']
    se_size_list = [3, 5, 10, 15]
    se_size_closing = 5

    # thick_type_list = ['closing_dilate']
    # se_size_list = [5]
    # se_size_closing = 5

    for thick_type in tqdm(thick_type_list, desc='thick type'):
        for se_size in tqdm(se_size_list, desc='se size'):
            for img_path in tqdm(img_path_list, desc='img path'):
                process_img(img_path.as_posix(), thick_type, se_size, se_size_closing, display)

    pass

if __name__ == '__main__':

    # main()
    # main_img_list()
    main_root_dir()

    pass