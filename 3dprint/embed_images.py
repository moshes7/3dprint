import numpy as np
import cv2
from pathlib import Path


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def resize_by_larger_dim(img, w_ref=1024, h_ref=1024, display=False):

    h, w = img.shape[0:2]

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
        cv2.destroyAllWindows()

    return img_resized

def transparent_background(img, display=False):

    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    alpha = np.sum(bgr, axis=-1) == 0
    alpha = np.uint8(alpha * 255)
    result = np.dstack([bgr, alpha])  # Add the mask as alpha channel

    if display:
        cv2.imshow('transparent', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result

def erode_img(img, se_size=5, display=False):

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))
    eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, se)

    if display:
        cv2.imshow('original', img)
        cv2.imshow('eroded', eroded)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return eroded


def inverse_img(img, display=False):

    inverse = 255 - img

    if display:
        cv2.imshow('inverse', inverse)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return inverse

def embed_single_line_on_background(img_file, background_file,
                                    size_wh_wanted=(1024, 1024), left_top=(1600, 750),
                                    output_subdir='',
                                    display=0):

    # read images
    img_orig = cv2.imread(img_file, 0)
    bg = cv2.imread(background_file, cv2.IMREAD_COLOR)

    # resize img
    img = resize_by_larger_dim(img_orig, w_ref=size_wh_wanted[0], h_ref=size_wh_wanted[1], display=display>3)

    # remove img background
    img = inverse_img(img, display=display>0)
    img = transparent_background(img, display>0)
    img = inverse_img(img, display=display>0)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    # embed img on cropped background
    x_start = left_top[0]
    x_end = x_start + img.shape[1]
    y_start = left_top[1]
    y_end = y_start + img.shape[0]
    bg_crop = bg[y_start:y_end, x_start:x_end]

    b_channel, g_channel, r_channel = cv2.split(bg_crop)
    img_on_bg_cropped = cv2.merge((b_channel, g_channel, r_channel, img))

    # put cropped background in full background
    img_on_bg = cv2.cvtColor(bg, cv2.COLOR_BGR2BGRA)
    img_on_bg[y_start:y_end, x_start:x_end, ...] = img_on_bg_cropped

    # img_on_bg = cv2.cvtColor(img_on_bg, cv2.COLOR_BGRA2BGR)

    # save result
    output_dir = Path(img_file).parent / 'output' / output_subdir
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file_name = '{}_{}.png'.format(Path(background_file).stem, Path(img_file).stem)
    output_file = output_dir / output_file_name
    cv2.imwrite(output_file.as_posix(), img_on_bg)

    if display > 0:
        cv2.imshow('original', img_orig)
        cv2.imshow('transparent', img)
        cv2.imshow('img_on_bg_cropped', img_on_bg_cropped)
        cv2.imshow('img_on_bg', img_on_bg)
        cv2.waitKey(0)

    return img_on_bg


def example_embed_single_line_on_background():

    img_file_list = [
        'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/rabbit.jpeg',
        'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/teddy_bear.jpeg',
        'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/mother_and_child.jpeg',
    ]

    background_file_list = [
        'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/background_1.jpg',
        'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/background_2.jpg',
        'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/background_3.jpg',
    ]

    display = 0
    output_subdir = '1_baseline'
    # output_subdir = '2_convert_to_bgr'
    # output_subdir = '3_jpg'

    for img_file in img_file_list:
        for background_file in background_file_list:

            embed_single_line_on_background(img_file, background_file,
                                            size_wh_wanted=(1024, 1024),
                                            left_top=(1600, 750),
                                            output_subdir=output_subdir,
                                            display=display)

    pass


def playground_embed_singeline_between_fingers():

    hand_1_file = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/hand_1.png'
    hand_2_file = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/hand_2.png'

    img_file = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/rabbit.jpeg'

    display = 0
    output_subdir = '4_hand_baseline'

    # put image on hand 1
    img_on_hand_1 = embed_single_line_on_background(img_file, hand_1_file,
                                                    size_wh_wanted=(200, 200), left_top=(250, 30),
                                                    output_subdir=output_subdir, display=display)

    # put hand 2 on img
    hand_2 = cv2.imread(hand_2_file, 0)
    hand_2 = cv2.cvtColor(hand_2, cv2.COLOR_BGR2BGRA)
    # hand_2_on_img = cv2.addWeighted(img_on_hand_1, 0.5, hand_2, 0.5, 0)
    hand_2_on_img = (0.5 * img_on_hand_1 + 0.5 * hand_2).astype(np.uint8)

    if display > 0:
        cv2.imshow('img_on_hand_1', img_on_hand_1)
        cv2.imshow('hand_2_on_img', hand_2_on_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    output_dir = Path(img_file).parent / 'output' / output_subdir
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file_name = '{}_on_hand_1.png'.format(Path(img_file).stem)
    output_file = output_dir / output_file_name
    cv2.imwrite(output_file.as_posix(), img_on_hand_1)
    output_file_name = 'hand_2_on_{}.png'.format(Path(img_file).stem)
    output_file = output_dir / output_file_name
    cv2.imwrite(output_file.as_posix(), hand_2)

    pass

def playground_2_embed_singeline_between_fingers():

    """
    Adapted from:
    https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_core/py_image_arithmetics/py_image_arithmetics.html
    """

    hand_1_file = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/hand_1.png'
    hand_2_file = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/hand_2.png'
    img_file = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/rabbit.jpeg'
    img_file = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/princess_and_butterfly.png'
    img_file = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/teddy_bear.jpeg'

    display = 1

    hand_1 = cv2.imread(hand_1_file, cv2.IMREAD_COLOR)
    hand_2 = cv2.imread(hand_2_file, cv2.IMREAD_COLOR)
    img_orig = cv2.imread(img_file, cv2.IMREAD_COLOR)

    # hand 1
    # --------
    # get masks
    img2gray = cv2.cvtColor(hand_1, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    if display > 1:
        cv2.imshow('hand_1', hand_1)
        cv2.imshow('img2gray', img2gray)
        cv2.imshow('ret', ret)
        cv2.imshow('mask', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # create white image
    out_img = np.zeros(hand_1.shape, dtype=np.uint8)
    out_img.fill(255)

    # black-out the area of logo in ROI
    rows, cols, channels = hand_1.shape
    roi = hand_1[0:rows, 0:cols ]
    hand_1_fg = cv2.bitwise_and(roi, roi, mask=mask)

    # dst = cv2.add(out_img, hand_1_fg)
    dst1 = out_img.copy()
    r, c = np.where(mask==(255))
    dst1[r, c, :] = hand_1_fg[r, c, :]

    if display > 1:
        cv2.imshow('dst1', dst1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # add shadow to img, inverse images for combining them properly
    blur_amount = 32
    img_orig_inverse = inverse_img(img_orig)
    img_blurred_inverse = cv2.blur(img_orig_inverse, (blur_amount, blur_amount))
    img_with_shadow_inverse = cv2.bitwise_or(img_orig_inverse, img_blurred_inverse)
    img_with_shadow = inverse_img(img_with_shadow_inverse)

    if display > 1:
        cv2.imshow('img_orig', img_orig)
        cv2.imshow('img_orig_inverse', img_orig_inverse)
        cv2.imshow('img_blurred_inverse', img_blurred_inverse)
        cv2.imshow('img_with_shadow_inverse', img_with_shadow_inverse)
        cv2.imshow('img_with_shadow', img_with_shadow)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # img
    # --------
    img = resize_by_larger_dim(img_with_shadow, w_ref=200, h_ref=200, display=display>3)
    img = inverse_img(img, display=False)
    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    img = inverse_img(img, display=False)

    if display > 1:
        cv2.imshow('img', img)
        cv2.imshow('img2gray', img2gray)
        cv2.imshow('ret', ret)
        cv2.imshow('mask', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # black-out the area of logo in ROI
    rows, cols, channels = img.shape
    roi = img[0:rows, 0:cols ]
    img_fg = cv2.bitwise_and(roi, roi, mask=mask)
    # img_fg = inverse_img(img_fg)

    # dst2 = cv2.add(dst, hand_2_fg)
    r, c = np.where(mask==(255))
    dst2 = dst1.copy()
    left = 250
    top = 55
    r_out = np.clip(r + top, a_min=0, a_max=dst2.shape[0]-1)
    c_out = np.clip(c + left, a_min=0, a_max=dst2.shape[1]-1)
    dst2[r_out, c_out, :] = img_fg[r, c, :]

    if display > 1:
        cv2.imshow('img_fg', img_fg)
        cv2.imshow('dst2', dst2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # hand 2
    # --------
    img2gray = cv2.cvtColor(hand_2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)

    if display > 1:
        cv2.imshow('hand_2', hand_2)
        cv2.imshow('img2gray', img2gray)
        cv2.imshow('ret', ret)
        cv2.imshow('mask', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # black-out the area of logo in ROI
    rows, cols, channels = hand_2.shape
    roi = hand_2[0:rows, 0:cols ]
    hand_2_fg = cv2.bitwise_and(roi, roi, mask=mask)

    # dst2 = cv2.add(dst, hand_2_fg)
    r, c = np.where(mask==(255))
    dst3 = dst2.copy()
    left = 0
    top = 25
    r_out = np.clip(r + top, a_min=0, a_max=dst3.shape[0]-1)
    c_out = np.clip(c + left, a_min=0, a_max=dst3.shape[1]-1)
    dst3[r_out, c_out, :] = hand_2_fg[r, c, :]

    if display > 1:
        cv2.imshow('dst3', dst3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    output_subdir = '6_hand_with_shadow_large_white_margins'
    output_dir = Path(img_file).parent / 'output' / output_subdir
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file_name = '{}_between_hands.png'.format(Path(img_file).stem)
    output_file = output_dir / output_file_name
    cv2.imwrite(output_file.as_posix(), dst3)

    pass


def playground_3_embed_singleline_between_fingers():

    # hand_1_file = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/hand_1.png'
    # hand_2_file = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/hand_2.png'
    hand_1_file = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/hand_bottom_left_1.png'
    hand_2_file = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/hand_bottom_left_2.png'

    img_file = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/rabbit.jpeg'
    # img_file = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/princess_and_butterfly.png'
    # img_file = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/teddy_bear.jpeg'
    # img_file = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/mother_and_child.jpeg'

    output_subdir = '10_with_shadow_average'

    # parameters
    display = 0
    blur_amount = 32
    th_gray = 150

    hand_1 = cv2.imread(hand_1_file, cv2.IMREAD_COLOR)
    hand_2 = cv2.imread(hand_2_file, cv2.IMREAD_COLOR)
    singleline_orig = cv2.imread(img_file, cv2.IMREAD_COLOR)

    # set background image
    bg_shape = (1600, 1200, 3)
    bg = np.zeros(bg_shape, dtype=np.uint8)
    bg.fill(255)  # white background

    # draw single-line shadow
    shadow_singleline = generate_shadow(singleline_orig, blur_amount=blur_amount, display=display>0)
    top_left_shadow_1 = (280, -50)
    resize_shadow_1 = (1200, 1200)
    img_with_shadow_1 = add_images(bg=bg, fg=shadow_singleline, fg_resize=resize_shadow_1, top_left=top_left_shadow_1, inverse_fg=True, display=display>0)

    # erode hands images - to delete white margins
    hand_1 = erode_img(hand_1, display=display>0)
    hand_2 = erode_img(hand_2, display=display>0)

    # draw hand 1 shadow
    blur_amount_2 = blur_amount
    shadow_hand = generate_shadow(hand_1, blur_amount=blur_amount_2, generate_mask=True, display=display>0)
    top = img_with_shadow_1.shape[0] - hand_1.shape[0]
    left = -80
    top_left_shadow_2 = (top, left)
    # resize_shadow_2 = None  #(1200, 1200)
    img_with_shadow_2 = add_shadows(img_with_shadow_1, shadow_hand, top_left=top_left_shadow_2, th_gray=10, addition_type='maximum', display=display>0)

    # add foreground of hand 1
    shift_shadow_hand = -5
    top_left_hand_1 = (top_left_shadow_2[0] - shift_shadow_hand, top_left_shadow_2[1] - shift_shadow_hand)
    img_with_hand_1 = add_images(bg=img_with_shadow_2, fg=hand_1, fg_resize=None, top_left=top_left_hand_1, inverse_fg=False, display=display>0)

    # add single-line
    shift_shadow = -30
    top_left_singleline = (top_left_shadow_1[0] - shift_shadow, top_left_shadow_1[1] - shift_shadow)
    img_with_singleline = add_images(bg=img_with_hand_1, fg=singleline_orig, fg_resize=resize_shadow_1, top_left=top_left_singleline,
                                     inverse_fg=True, th_gray=th_gray, display=display>0)

    # add foreground of hand 2
    top = img_with_singleline.shape[0] - hand_2.shape[0]
    left = -80
    img_with_hand_2 = add_images(bg=img_with_singleline, fg=hand_2, fg_resize=None, top_left=(top, left), inverse_fg=False, display=display>0)

    out_img = img_with_hand_2

    output_dir = Path(img_file).parent / 'output' / output_subdir
    output_dir.mkdir(exist_ok=True, parents=True)
    output_file_name = '{}_between_hands.png'.format(Path(img_file).stem)
    output_file = output_dir / output_file_name
    cv2.imwrite(output_file.as_posix(), out_img)

    pass


def generate_shadow(img, blur_amount=48, generate_mask=False, display=False):

    if generate_mask:
        # inverse_fg = True
        th_gray = 10
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gray = inverse_img(gray) if inverse_fg else gray
        ret, mask = cv2.threshold(gray, th_gray, 255, cv2.THRESH_BINARY)
        mask_inverse = inverse_img(mask)
        img = cv2.cvtColor(mask_inverse, cv2.COLOR_GRAY2BGR)

    shadow = cv2.blur(img, (blur_amount, blur_amount))

    if display:
        if generate_mask:
            cv2.imshow('gray', gray)
            cv2.imshow('mask', mask)
            cv2.imshow('mask_inverse', mask_inverse)
        cv2.imshow('img', img)
        cv2.imshow('shadow', shadow)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return shadow

def add_shadows(shadow_1, shadow_2, top_left=(0, 0), th_gray=150, addition_type='maximum', display=False):

    # inverse
    shadow_1 = inverse_img(shadow_1)
    shadow_2 = inverse_img(shadow_2)

    shadow_2 = erode_img(shadow_2)

    ret, mask = cv2.threshold(shadow_2, th_gray, 255, cv2.THRESH_BINARY)

    out_inverse = shadow_1.copy()
    r, c = np.where(mask == 255)[:2]
    top = top_left[0]
    left = top_left[1]
    r_out = np.clip(r + top, a_min=0, a_max=out_inverse.shape[0]-1)
    c_out = np.clip(c + left, a_min=0, a_max=out_inverse.shape[1]-1)
    if addition_type == 'maximum':  # take maximum between shadow_1 and shadow_2
        # out_inverse[r_out, c_out, :] = cv2.bitwise_or(out_inverse[r_out, c_out, :], shadow_2[r, c, :])
        out_inverse[r_out, c_out, :] = np.maximum(out_inverse[r_out, c_out, :], shadow_2[r, c, :])
    elif addition_type == 'masked_addition':  # use shadow_2 values as is where they occur
        # shadow_2 = cv2.bitwise_and(shadow_2, shadow_2, mask)
        out_inverse[r_out, c_out, :] = shadow_2[r, c, :]

    out = inverse_img(out_inverse)

    if display:
        cv2.imshow('shadow_1', shadow_1)
        cv2.imshow('shadow_2', shadow_2)
        cv2.imshow('mask', mask)
        cv2.imshow('out_inverse', out_inverse)
        cv2.imshow('out', out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return out

def add_images(bg, fg, fg_resize=None, top_left=(0, 0), inverse_fg=False, th_gray=10, display=False):

    # resize
    fg = resize_by_larger_dim(fg, w_ref=fg_resize[0], h_ref=fg_resize[1], display=False) if fg_resize is not None else fg

    # get foreground mask
    gray = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
    gray = inverse_img(gray) if inverse_fg else gray
    ret, mask = cv2.threshold(gray, th_gray, 255, cv2.THRESH_BINARY)

    # extract foreground using mask
    fg_masked = cv2.bitwise_and(fg, fg, mask)

    # add images
    out = bg.copy()
    r, c = np.where(mask == 255)
    top = top_left[0]
    left = top_left[1]
    r_out = np.clip(r + top, a_min=0, a_max=out.shape[0]-1)
    c_out = np.clip(c + left, a_min=0, a_max=out.shape[1]-1)
    out[r_out, c_out, :] = fg_masked[r, c, :]

    if display:
        cv2.imshow('bg', bg)
        cv2.imshow('gray', gray)
        # cv2.imshow('fg', fg)
        # cv2.imshow('mask', mask)
        cv2.imshow('fg_masked', fg_masked)
        cv2.imshow('out', out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return out

def find_singleline_bottom_left_example():

    # img_file = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/rabbit.jpeg'
    # img_file = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/princess_and_butterfly.png'
    # img_file = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/teddy_bear.jpeg'
    img_file = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/mother_and_child.jpeg'

    display = True

    singleline_orig = cv2.imread(img_file, cv2.IMREAD_COLOR)
    fg = singleline_orig

    fg_resize = (1200, 1200)
    fg = resize_by_larger_dim(fg, w_ref=fg_resize[0], h_ref=fg_resize[1], display=False) if fg_resize is not None else fg

    left, bottom = find_singleline_bottom_left(fg, th_gray=10, inverse=True, display=display)

    pass

def find_singleline_bottom_left(img, th_gray=10, inverse=True, display=False):

    # get foreground mask
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = inverse_img(gray) if inverse else gray
    ret, mask = cv2.threshold(gray, th_gray, 255, cv2.THRESH_BINARY)

    # find bottom left point
    bottom_inds = last_nonzero(mask, axis=0, invalid_val=-1)
    bottom_inds_non_zero = np.where(bottom_inds > 0)[0]
    bottom_inds = bottom_inds[bottom_inds_non_zero]
    width = bottom_inds_non_zero.max() - bottom_inds_non_zero.min()
    bottom_ind = int(0.1 * width)  # take point 10% inside the single-line
    bottom = bottom_inds[bottom_ind]
    left = bottom_inds_non_zero[bottom_ind]

    if display:
        img_with_circle = cv2.circle(img, (left, bottom), 10, (0, 0, 255), -1)
        cv2.imshow('img_with_circle', img_with_circle)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return left, bottom

"""
Next 2 function were taken from:
https://stackoverflow.com/questions/66440022/get-non-zero-roi-from-numpy-array  # last answer
https://stackoverflow.com/questions/47269390/how-to-find-first-non-zero-value-in-every-column-of-a-numpy-array
"""
def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    ind = np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)
    return ind

def last_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    ind = np.where(mask.any(axis=axis), val, invalid_val)
    return ind


def find_thumb_center():

    hand_1_file = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/hand_bottom_left_1.png'
    hand_2_file = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/hand_bottom_left_2.png'
    hand_1 = cv2.imread(hand_1_file, cv2.IMREAD_COLOR)
    hand_2 = cv2.imread(hand_2_file, cv2.IMREAD_COLOR)

    left = 300
    bottom = 230

    hand_2_with_circle = cv2.circle(hand_2, (left, bottom), 10, (0, 0, 255), -1)

    cv2.imshow('hand_2_with_circle', hand_2_with_circle)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    pass

if __name__ == '__main__':

    # example_embed_single_line_on_background()
    # playground_embed_singeline_between_fingers()
    # playground_2_embed_singeline_between_fingers()
    # playground_3_embed_singleline_between_fingers()
    find_singleline_bottom_left_example()
    # find_thumb_center()


    pass