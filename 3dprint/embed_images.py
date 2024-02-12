import numpy as np
import cv2
from pathlib import Path
from rembg import remove

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


def resize_by_larger_dim(img, w_ref=1024, h_ref=1024, display=False):

    h, w = img.shape

    if w >= h:
        width = w_ref
        height = None
        interp = cv2.INTER_AREA if w > w_ref else cv2.INTER_CUBIC  # AREA for shrinking, CUBIC for enlarging
    else:
        width = None
        height = h_ref
        interp = cv2.INTER_AREA if h > h_ref else cv2.INTER_CUBIC

    img_resized = resize(img, width, height, interp)

    if display:
        cv2.imshow('Input', img)
        cv2.imshow('Resized', img_resized)
        cv2.waitKey(0)

    return img_resized

def transparent_background(img, display=False):

    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    alpha = np.sum(bgr, axis=-1) == 0
    alpha = np.uint8(alpha * 255)
    result = np.dstack([bgr, alpha])  # Add the mask as alpha channel

    if display:
        cv2.imshow('transparent', result)
        cv2.waitKey(0)

    return result


def inverse_img(img, display=False):

    inverse = 255 - img

    if display:
        cv2.imshow('inverse', inverse)
        cv2.waitKey(0)

    return inverse

def embed_single_line_on_background():

    # img_file = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/rabbit.jpeg'
    img_file = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/teddy_bear.jpeg'
    img_file = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/mother_and_child.jpeg'
    img_file = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/princess_and_butterfly.png'
    background_file = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/background_1.jpg'
    # background_file = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/background_2.jpg'
    # background_file = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/background_3.jpg'

    display = 1


    # load images
    img_orig = cv2.imread(img_file, 0)
    bg = cv2.imread(background_file, 0)

    # bg = cv2.cvtColor(bg, cv2.COLOR_BGR2BGRA)
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)

    img = resize_by_larger_dim(img_orig, w_ref=1024, h_ref=1024, display=display>3)


    # remove img
    img = inverse_img(img, display=display>2)
    img = transparent_background(img, display>2)
    # img = remove(img)
    img = inverse_img(img, display=display>2)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    if display > 2:
        cv2.imshow('original', img_orig)
        cv2.imshow('transparent', img)
        cv2.waitKey(0)

    # embed img on background
    # adapted from: https://stackoverflow.com/questions/69620706/overlay-image-on-another-image-with-opencv-and-numpy
    # extract alpha channel from foreground image as mask and make 3 channels
    # alpha = img[:,:,3]
    # alpha = cv2.merge([alpha,alpha,alpha])
    #
    # # extract bgr channels from foreground image
    # front = img[:,:,0:3]
    #
    # blend the two images using the alpha channel as controlling mask
    # result = np.where(alpha==(0,0,0), bg, front)

    left_top = (1600, 750)
    x_start = left_top[0]
    x_end = x_start + img.shape[1]
    y_start = left_top[1]
    y_end = y_start + img.shape[0]
    bg_crop = bg[y_start:y_end, x_start:x_end]

    # img_on_bg_cropped = cv2.addWeighted(bg_crop, 1, img, 1, 0)
    # img_on_bg_cropped = np.where(alpha<=(20,20,20), bg_crop, front)

    b_channel, g_channel, r_channel = cv2.split(bg_crop)
    img_on_bg_cropped = cv2.merge((b_channel, g_channel, r_channel, img))


    # show result
    # img_on_bg = bg.copy()
    img_on_bg = cv2.cvtColor(bg, cv2.COLOR_BGR2BGRA)
    img_on_bg[y_start:y_end, x_start:x_end, ...] = img_on_bg_cropped
    # cv2.imshow("img_on_bg", img_on_bg)
    # cv2.waitKey(0)

    output_dir = Path(img_file).parent / 'output'
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file_name = '{}_{}.png'.format(Path(background_file).stem, Path(img_file).stem)
    output_file = output_dir / output_file_name
    cv2.imwrite(output_file.as_posix(), img_on_bg)

    pass


if __name__ == '__main__':

    embed_single_line_on_background()

    pass