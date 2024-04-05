from pathlib import Path

from print3d import embed_images_wrapper

def example_embed_images_wrapper():

    singleline_file = Path(__file__).parents[1] / 'images' / 'singlelines' / 'mother_and_child.jpeg'
    output_dir = Path(__file__).parents[3] / 'output' / 'example_embed_images_wrapper'

    img_out_type = 'jpg'  # or 'png'

    background_file_list = [
        (Path(__file__).parents[1] / 'images' / 'backgrounds' / 'background_1.jpg').as_posix(),
        (Path(__file__).parents[1] / 'images' / 'backgrounds' / 'background_2.jpg').as_posix(),
        (Path(__file__).parents[1] / 'images' / 'backgrounds' / 'background_3.jpg').as_posix(),
    ]

    bg_size_wh_wanted_list = [(1024, 1024), (1024, 1024), (1200, 1200)]
    bg_top_left_list = [(1550, 750), (1900, 500), (1100, 1000)]

    singleline_file = singleline_file.as_posix()  # convert to str

    out_file_list = embed_images_wrapper(singleline_file, background_file_list, output_dir,
                                         display=0,
                                         img_out_type=img_out_type,
                                         finger_params={'resize_singleline': (750, 750),
                                                        'out_img_shape': (1000, 1200, 3),
                                                        'hand_type_list': ['bottom_left', 'center'],
                                                        },
                                         background_params={'size_wh_wanted': bg_size_wh_wanted_list,
                                                            'left_top': bg_top_left_list,
                                                            }
                                         )

    print(out_file_list)

    pass


if __name__ == '__main__':

    example_embed_images_wrapper()

    pass