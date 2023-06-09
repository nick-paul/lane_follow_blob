from numpy import ndarray
import cv2 as cv
import numpy as np
from lane_follow_blob.utils import cols, rows
from lane_follow_blob.cfg import BlobConfig

class EdgeMethod:
    """ Must match edge_method enum in config """
    canny = 0
    laplac = 1


def compute_lines(image: ndarray,
                  config: BlobConfig,
                  debug_image=None) -> ndarray:
    lines_mat = np.zeros_like(image)

    # Images to draw lines on 
    if debug_image is None:
        draw_lines_on = (lines_mat,)
    else:
        draw_lines_on = (lines_mat, debug_image)

    # Crop out the upper portion of the image
    # This helps with avoiding detecting lines
    # found in objects along the horizon (i.e. trees, posts)
    x = 0
    y = int(config.lines_top * rows(image))
    w = cols(image)
    h = int(rows(image) - config.lines_top * rows(image))
    image_cropped = image[y:y+h, x:x+w]

    lines = cv.HoughLinesP(image_cropped,
                            rho=config.lines_rho,
                            theta=0.01745329251,
                            threshold=config.lines_thresh,
                            minLineLength=config.lines_min_len,
                            maxLineGap=config.lines_max_gap)
    if lines is not None:
        for l in lines:
            l = l[0] # (4,1) => (4,)
            diffx = l[0] - l[2]
            diffy = l[1] - l[3]

            slope = diffy / diffx

            if abs(slope) < config.lines_min_slope or abs(slope) > config.lines_max_slope:
                continue

            diffx *= config.lines_extend
            diffy *= config.lines_extend

            l[0] -= diffx
            l[1] -= diffy
            l[2] += diffx
            l[3] += diffy

            for img in draw_lines_on:
                cv.line(img,
                    (l[0], int(l[1] + config.lines_top * rows(image))),
                    (l[2], int(l[3] + config.lines_top * rows(image))),
                    255, 5)

    return lines_mat



def find_lanes(input_image: ndarray,
               config: BlobConfig,
               debug_image: ndarray=None) -> ndarray:
    """
    This algorithm uses light-on-dark contrast to find 
    lane lines. If lanes do not have this property, another
    lane-finding algorithm may be used instead
    """

    image = input_image

    # Median blur
    if config.med_blur_enable:
        image = cv.medianBlur(image, config.med_blur * 2 + 1)

    if config.gauss_blur_enable:
        gauss_k = config.gauss_blur * 2 + 1
        image = cv.GaussianBlur(image,(gauss_k, gauss_k),cv.BORDER_DEFAULT)

    if config.enable_less_color:
        image = cv.cvtColor(image, cv.COLOR_BGR2HSV);
        channels = cv.split(image)
        channels[2] -= (config.less_color_mux * channels[1]).astype(np.uint8)
        image = cv.merge(channels);
        image = cv.cvtColor(image, cv.COLOR_HSV2BGR);

    if config.enable_clahe:
        # convert to lab
        lab = cv.split(cv.cvtColor(image, cv.COLOR_BGR2LAB))
        clahe = cv.createCLAHE(clipLimit=config.clahe_clip)
        lab[0] = clahe.apply(lab[0]) # apply to lum channel only
        image = cv.cvtColor(cv.merge(lab), cv.COLOR_LAB2BGR)

    if config.enable_color_correct:
        image = cv.convertScaleAbs(image, alpha=config.cc_alpha, beta=config.cc_beta)

    if config.enable_sharpen:
        image2 = cv.GaussianBlur(image, (0, 0), config.sharp_kernel * 2 + 1);
        image = cv.addWeighted(image2, 1 + config.sharp_weight, image, -1 * config.sharp_weight, 0);


    if config.edge_detect_enable:
        if config.edge_method == EdgeMethod.canny:
            image = cv.Canny(image, config.canny_lower_thresh,
                    config.canny_upper_thresh,
                    apertureSize=config.canny_aperture_size * 2 + 1)
        elif config.edge_method == EdgeMethod.laplac:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            image = cv.Laplacian(image,cv.CV_64F, config.lapla_ksize * 2 + 1)
            _,image = cv.threshold(image,0,255.0,cv.THRESH_TOZERO)
            image = np.uint8(image)
            #image = cv.Sobel(image, -1, config.sobel_xorder,
            #            config.sobel_yorder,
            #            config.sobel_ksize * 2 + 1)
            if config.lapla_thresh_enable:
                _, image = cv.threshold(image,config.lapla_thresh,255,cv.THRESH_BINARY) 

    # Dilate images
    if config.dilation_enable:
        dilation_size = (2 * config.dilation_size + 1, 2 * config.dilation_size + 1)
        dilation_anchor = (config.dilation_size, config.dilation_size)
        dilate_element = cv.getStructuringElement(cv.MORPH_RECT, dilation_size, dilation_anchor)
        image = cv.dilate(image, dilate_element)

    if config.lines_enable:
        image = compute_lines(image, config, debug_image=debug_image)

    # always at least threshold it since the output is expecting a black and white image

    return image