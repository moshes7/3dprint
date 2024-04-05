"""
Code adapted from:
https://withoutbg.com/resources/adding-drop-shadow
"""

import cv2
import numpy as np


def load_image(path, color_conversion=None):
    """Load an image and optionally convert its color."""
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if color_conversion:
        image = cv2.cvtColor(image, color_conversion)
    return image

def extract_alpha_channel(image):
    """Extract the alpha channel and the RGB channels from an image."""
    alpha_channel = image[:,:,3]
    rgb_channels = image[:,:,0:3]
    return alpha_channel, rgb_channels

def apply_blur_to_alpha(alpha, blur_amount):
    """Apply blur to the alpha channel."""
    return cv2.blur(alpha, (blur_amount, blur_amount))

def expand_and_normalize_alpha(alpha):
    """Expand alpha dimensions and normalize its values to the range [0,1]."""
    expanded_alpha = np.expand_dims(alpha, axis=2)
    repeated_alpha = np.repeat(expanded_alpha, 3, axis=2)
    normalized_alpha = repeated_alpha / 255
    return normalized_alpha

def create_shadow_on_bg(bg, alpha_blur):
    """Put shadow (based on blurred alpha) on top of the background."""
    black_canvas = np.zeros(bg.shape, dtype=np.uint8)
    shadowed_bg = (alpha_blur * black_canvas + (1 - alpha_blur) * bg).astype(np.uint8)
    return shadowed_bg

def composite_foreground_on_bg(fg, alpha, bg_with_shadow):
    """Put the foreground image on top of the background with shadow."""
    composited_image = (alpha * fg + (1 - alpha) * bg_with_shadow).astype(np.uint8)
    return composited_image

def inverse_img(img, display=False):

    inverse = 255 - img

    if display:
        cv2.imshow('inverse', inverse)
        cv2.waitKey(0)

    return inverse


if __name__ == "__main__":

    FG_IMG_PATH = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/rabbit.jpeg'
    BG_IMG_PATH = 'C:/Users/Moshe/Sync/Projects/3d_printing/images/backgrounds/background_1.jpg'
    BLUR_AMOUNT = 32
    display = 1

    # Load images and convert their color if necessary
    fg = load_image(FG_IMG_PATH, cv2.COLOR_BGRA2RGBA)
    bg = load_image(BG_IMG_PATH, cv2.COLOR_RGB2BGR)

    left_top=(1600, 750)
    x_start = left_top[0]
    x_end = x_start + fg.shape[1]
    y_start = left_top[1]
    y_end = y_start + fg.shape[0]
    bg_crop = bg[y_start:y_end, x_start:x_end]

    # Extract alpha and RGB channels from the foreground image
    alpha_orig, fg_rgb = extract_alpha_channel(fg)
    alpha = cv2.cvtColor(fg_rgb, cv2.COLOR_BGR2GRAY)

    if display > 1:
        cv2.imshow("alpha_orig", alpha_orig)
        cv2.imshow("fg_rgb", fg_rgb)
        cv2.imshow("alpha", alpha)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Blur the alpha channel to get the shadow
    alpha_blur = apply_blur_to_alpha(alpha, BLUR_AMOUNT)

    # Expand and normalize the blurred alpha for shadow calculation
    alpha_blur_normalized = expand_and_normalize_alpha(alpha_blur)

    if display > 1:
        cv2.imshow("alpha", alpha)
        cv2.imshow("alpha_blur", alpha_blur)
        cv2.imshow("alpha_blur_normalized", alpha_blur_normalized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Create a version of the background with the shadow
    bg_with_shadow = create_shadow_on_bg(bg_crop, alpha_blur_normalized)
    bg_with_shadow_inverse = inverse_img(bg_with_shadow, display=False)

    # Expand and normalize the original alpha for compositing foreground over background
    alpha_normalized = expand_and_normalize_alpha(alpha)

    if display > 1:
        cv2.imshow("bg_with_shadow", bg_with_shadow)
        cv2.imshow("bg_with_shadow_inverse", bg_with_shadow_inverse)
        cv2.imshow("alpha_normalized", alpha_normalized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Composite the foreground on the shadowed background
    final_image = composite_foreground_on_bg(fg_rgb, alpha_normalized, bg_with_shadow_inverse)

    if display > 1:
        cv2.imshow("fg_rgb", fg_rgb)
        cv2.imshow("alpha_normalized", alpha_normalized)
        cv2.imshow("bg_with_shadow_inverse", bg_with_shadow_inverse)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Display the final image (optional)
    if display > 1:
        cv2.imshow("Final Image", final_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    alpha_inverse = inverse_img(alpha)
    bg_with_shadow_gray = cv2.cvtColor(bg_with_shadow, cv2.COLOR_BGR2GRAY)
    # img_with_shadow_inverse = np.clip(alpha_inverse + bg_with_shadow_gray, 0, 255).astype(np.uint8)
    img_with_shadow_inverse = cv2.bitwise_or(alpha_inverse, bg_with_shadow_gray)
    img_with_shadow = inverse_img(img_with_shadow_inverse)

    if display > 0:
        cv2.imshow("alpha_inverse", alpha_inverse)
        cv2.imshow("bg_with_shadow", bg_with_shadow)
        cv2.imshow("img_with_shadow_inverse", img_with_shadow_inverse)
        cv2.imshow("img_with_shadow", img_with_shadow)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    pass



