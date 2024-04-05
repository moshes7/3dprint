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


def embed_single_line_on_background(singleline_img_or_file, background_file,
                                    size_wh_wanted=(1024, 1024), left_top=(1600, 750),
                                    out_file_name=None,
                                    display=0):
    """
        Embed single-line image on background image.

        Parameters
        ----------
        singleline_img_or_file : ndarray or str
            Single-line image, may be the image itself, or it's full path.
        background_file : ndarray or str
            Single-line image, may be the image itself, or it's full path.
        size_wh_wanted : tuple, optional
            Wanted single-line size.
       left_top : tuple, optional
            Wanted single-line top-left coordinate.
        display : int, optional
            Display flag:
                - 0: no display
                - 1: display output image only
                - 2: display all intermediate images (for debug)
        out_file_name: str, optional
            Wanted output file name, saved only if not None.

        Returns
        -------
        out_img : ndarray
            Output image.
    """

    # read images
    bg = cv2.imread(background_file, cv2.IMREAD_COLOR)
    singleline_orig = cv2.imread(singleline_img_or_file, cv2.IMREAD_COLOR) if isinstance(singleline_img_or_file, str) else singleline_img_or_file
    singleline = resize_by_larger_dim(singleline_orig, w_ref=size_wh_wanted[0], h_ref=size_wh_wanted[1], display=False)

    # embed single-line on background
    out_img = add_images(bg=bg, fg=singleline, fg_resize=None, top_left=(left_top[1], left_top[0]), inverse_fg=True, display=display>1)

    # save output image
    if out_file_name is not None:
        out_file_name = Path(out_file_name)
        Path(out_file_name).parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(out_file_name.as_posix(), out_img)

    return out_img


def embed_singleline_between_fingers(singleline_img_or_file,
                                     resize_singleline=(750, 750),
                                     out_img_shape=(1000, 1200, 3),
                                     hand_type='bottom_left',
                                     singleline_shadow_shift=-25,
                                     hand_shadow_shift=-10,
                                     blur_amount=32,
                                     th_gray=150,
                                     display=0,
                                     out_file_name=None,
                                     ):
    """
        Embed single-line image between 2 hand images, and add shdows

        Parameters
        ----------
        singleline_img_or_file : ndarray or str
            Single-line image, may be the image itself, or it's full path.
        resize_singleline : tuple, optional
            Wanted single-line size.
        out_img_shape : tuple, optional
            Output image shape.
        hand_type : str, optional
            Hand images type, must be one of {'bottom_left', 'center'}.
        singleline_shadow_shift : int, optional
            Single-line shadow shift.
        hand_shadow_shift : int, optional
            Hand shadow shift.
        blur_amount : int, optional
            Controls the shadow intensity.
        th_gray : int, optional
            Threshold to convert grayscale image to binary one.
        display : int, optional
            Display flag:
                - 0: no display
                - 1: display output image only
                - 2: display all intermediate images (for debug)
        out_file_name: str, optional
            Wanted output file name, saved only if not None.

        Returns
        -------
        out_img : ndarray
            Output image.
    """

    assert isinstance(singleline_img_or_file, str) or isinstance(singleline_img_or_file, np.array)
    assert hand_type in ['bottom_left', 'center']

    if hand_type == 'bottom_left':
        hand_1_file = (Path(__file__).parent / 'images' / 'hands' / 'hand_bottom_left_1.png').as_posix()
        hand_2_file = (Path(__file__).parent / 'images' / 'hands' / 'hand_bottom_left_2.png').as_posix()
        thumb_center_orig_xy = (300, 230)  # values for hand_bottom_left_2.png, in original img coordinates
        hand_shift_top_left = (0, -580)
        width_percentage = 0.1  # where fingers will hold the single-line
    elif hand_type == 'center':
        hand_1_file = (Path(__file__).parent / 'images' / 'hands' / 'hand_center_1.png').as_posix()
        hand_2_file = (Path(__file__).parent / 'images' / 'hands' / 'hand_center_2.png').as_posix()
        thumb_center_orig_xy = (315, 210)  # values for hand_bottom_left_2.png, in original img coordinates
        hand_shift_top_left = (0, -300)
        width_percentage = 0.4  # where fingers will hold the single-line
        pass

    # read images
    hand_1 = cv2.imread(hand_1_file, cv2.IMREAD_COLOR)
    hand_2 = cv2.imread(hand_2_file, cv2.IMREAD_COLOR)
    singleline_orig = cv2.imread(singleline_img_or_file, cv2.IMREAD_COLOR) if isinstance(singleline_img_or_file, str) else singleline_img_or_file
    singleline = resize_by_larger_dim(singleline_orig, w_ref=resize_singleline[0], h_ref=resize_singleline[1], display=False)

    # set background image
    bg = np.zeros(out_img_shape, dtype=np.uint8)
    bg.fill(255)  # white background

    # set top-left position of different objects
    top = bg.shape[0] - hand_1.shape[0] + hand_shift_top_left[0]
    left = bg.shape[1] - singleline.shape[1] + hand_shift_top_left[1]
    top_left_hand = (top, left)

    # get singleline top left position
    left, bottom = find_singleline_bottom_left(singleline, th_gray=10, width_percentage=width_percentage, inverse=True, display=display>1)
    thumb_center_xy = (top_left_hand[1] + thumb_center_orig_xy[0], top_left_hand[0] + thumb_center_orig_xy[1])
    top_singleline = thumb_center_xy[1] - bottom
    left_singleline = thumb_center_xy[0] - left
    top_left_singleline = (top_singleline, left_singleline)

    # draw single-line shadow
    shadow_singleline = generate_shadow(singleline, blur_amount=blur_amount, display=display>1)
    shadow_shift = singleline_shadow_shift
    top_left_singleline_shadow = (top_left_singleline[0] - shadow_shift, top_left_singleline[1] - shadow_shift)
    img_with_shadow_1 = add_images(bg=bg, fg=shadow_singleline, fg_resize=None, top_left=top_left_singleline_shadow, inverse_fg=True, display=display>1)

    # erode hands images - to delete white margins
    hand_1 = erode_img(hand_1, display=display>1)
    hand_2 = erode_img(hand_2, display=display>1)

    # draw hand 1 shadow
    blur_amount_2 = blur_amount
    shadow_hand = generate_shadow(hand_1, blur_amount=blur_amount_2, generate_mask=True, display=display>1)
    shadow_shift = hand_shadow_shift
    top_left_hand_shadow = (top_left_hand[0] - shadow_shift, top_left_hand[1] - shadow_shift)
    img_with_shadow_2 = add_shadows(img_with_shadow_1, shadow_hand, top_left=top_left_hand_shadow, th_gray=10, addition_type='maximum', display=display>1)

    # add foreground of hand 1
    img_with_hand_1 = add_images(bg=img_with_shadow_2, fg=hand_1, fg_resize=None, top_left=top_left_hand, inverse_fg=False, display=display>1)

    # add single-line
    img_with_singleline = add_images(bg=img_with_hand_1, fg=singleline, fg_resize=None, top_left=top_left_singleline,
                                     inverse_fg=True, th_gray=th_gray, display=display>1)

    # add foreground of hand 2
    img_with_hand_2 = add_images(bg=img_with_singleline, fg=hand_2, fg_resize=None, top_left=top_left_hand, inverse_fg=False, display=display>0)

    out_img = img_with_hand_2

    if out_file_name is not None:
        out_file_name = Path(out_file_name)
        Path(out_file_name).parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(out_file_name.as_posix(), out_img)

    return out_img

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

def find_singleline_bottom_left(img, th_gray=10, width_percentage=0.1, inverse=True, display=False):

    # get foreground mask
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = inverse_img(gray) if inverse else gray
    ret, mask = cv2.threshold(gray, th_gray, 255, cv2.THRESH_BINARY)

    # find bottom left point
    bottom_inds = last_nonzero(mask, axis=0, invalid_val=-1)
    bottom_inds_non_zero = np.where(bottom_inds > 0)[0]
    bottom_inds = bottom_inds[bottom_inds_non_zero]
    width = bottom_inds_non_zero.max() - bottom_inds_non_zero.min()
    bottom_ind = int(width_percentage * width)  # take point 10% inside the single-line
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

    # bottom left
    hand_1_file = (Path(__file__).parent / 'images' / 'hand_bottom_left_1.png').as_posix()
    hand_2_file = (Path(__file__).parent / 'images' / 'hand_bottom_left_2.png').as_posix()
    hand_1 = cv2.imread(hand_1_file, cv2.IMREAD_COLOR)
    hand_2 = cv2.imread(hand_2_file, cv2.IMREAD_COLOR)

    left = 300
    bottom = 230

    hand_2_with_circle = cv2.circle(hand_2, (left, bottom), 10, (0, 0, 255), -1)

    cv2.imshow('hand_2_with_circle', hand_2_with_circle)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # center
    hand_1_file = (Path(__file__).parent / 'images' / 'hand_center_1.png').as_posix()
    hand_2_file = (Path(__file__).parent / 'images' / 'hand_center_2.png').as_posix()
    hand_1 = cv2.imread(hand_1_file, cv2.IMREAD_COLOR)
    hand_2 = cv2.imread(hand_2_file, cv2.IMREAD_COLOR)

    left = 315
    bottom = 210

    hand_2_with_circle = cv2.circle(hand_2, (left, bottom), 10, (0, 0, 255), -1)

    cv2.imshow('hand_2_with_circle', hand_2_with_circle)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    pass


def example_embed_singleline_between_fingres():

    singleline_file = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/mother_and_child.jpeg'
    # singleline_file = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/rabbit.jpeg'
    # singleline_file = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/princess_and_butterfly.png'
    # singleline_file = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/teddy_bear.jpeg'

    hand_type = 'bottom_left'
    # hand_type = 'center'

    output_root_dir = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/output/'
    output_subdir = '12_dedicated_function'
    out_file = '{}_between_hands_{}.png'.format(Path(singleline_file).stem, hand_type)

    out_file_name = Path(output_root_dir) / output_subdir / out_file

    embed_singleline_between_fingers(singleline_file,
                                     resize_singleline=(750, 750),
                                     out_img_shape=(1000, 1200, 3),
                                     hand_type=hand_type,
                                     singleline_shadow_shift=-25,
                                     hand_shadow_shift=-10,
                                     blur_amount=32,
                                     th_gray=150,
                                     display=0,
                                     out_file_name=out_file_name,
                                     )

    pass


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

    display = 1

    output_root_dir = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/output/'
    output_subdir = '13_improved_img_on_bg'

    for img_file in img_file_list:
        for background_file in background_file_list:

            out_file = '{}_on_{}.png'.format(Path(img_file).stem, Path(background_file).stem)
            out_file_name = Path(output_root_dir) / output_subdir / out_file

            embed_single_line_on_background(img_file, background_file,
                                            size_wh_wanted=(1024, 1024),
                                            left_top=(1600, 750),
                                            out_file_name=out_file_name,
                                            display=display)

    pass


def embed_images_wrapper(singleline_file, background_file_list, output_dir,
                          display=0,
                          finger_params={'resize_singleline': (750, 750),
                                         'out_img_shape': (1000, 1200, 3),
                                         'hand_type_list': ['bottom_left', 'center'],
                                         },
                          ):

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    output_file_list = []

    for hand_type in finger_params['hand_type_list']:
        output_file_name = output_dir / '{}_between_single_{}.png'.format(Path(singleline_file).stem, hand_type)
        output_file_list.append(output_file_name.as_posix())
        embed_singleline_between_fingers(singleline_file,
                                         resize_singleline=finger_params['resize_singleline'],
                                         out_img_shape=finger_params['out_img_shape'],
                                         hand_type=hand_type,
                                         singleline_shadow_shift=-25,
                                         hand_shadow_shift=-10,
                                         blur_amount=32,
                                         th_gray=150,
                                         display=display,
                                         out_file_name=output_file_name,
                                         )


    # for background_file in background_file_list:
    #     out_file = '{}_on_{}.png'.format(Path(img_file).stem, Path(background_file).stem)
    #     out_file_name = Path(output_root_dir) / output_subdir / out_file
    #
    #     embed_single_line_on_background(img_file, background_file,
    #                                     size_wh_wanted=(1024, 1024),
    #                                     left_top=(1600, 750),
    #                                     out_file_name=out_file_name,
    #                                     display=display)


    return output_file_list

def example_embed_images_wrapper():

    singleline_file = Path(__file__).parent / 'images' / 'singlelines' / 'mother_and_child.jpeg'
    output_dir = Path(__file__).parents[2] / 'output' / 'example_embed_images_wrapper'

    background_file_list = [
        'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/background_1.jpg',
        'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/background_2.jpg',
        'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/background_3.jpg',
    ]


    singleline_file = singleline_file.as_posix()  # convert to str

    output_file_list = embed_images_wrapper(singleline_file, background_file_list, output_dir,
                                            display=0,
                                            finger_params={'resize_singleline': (750, 750),
                                                           'out_img_shape': (1000, 1200, 3),
                                                           'hand_type_list': ['bottom_left', 'center'],
                                                           },
                                            )


    pass


if __name__ == '__main__':

    # find_singleline_bottom_left_example()
    # find_thumb_center()
    # example_embed_singleline_between_fingres()
    # example_embed_single_line_on_background()
    example_embed_images_wrapper()


    pass