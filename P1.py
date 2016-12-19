# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import numpy as np
import cv2
import os


global parameters
parameters = \
    {
        'debug': False,
        'output_filter': True,
        'plot_output': True,
        'cmap': 'gray',
        'test_image_index': -1,
        # convert to grayscale
        'grayscale': True,
        # smoothing
        'smoothing': True,
        'kernel_size': 5,
        # Canny
        'canny': True,
        'low_threshold': 50,        # 50
        'high_threshold': 150,      # 150
        # (normalized) RoI mask vertices
        'mask': True,
        'norm_vertices': np.array([[(0.05, 1), (0.95, 1), (0.51, 0.55), (0.49, 0.55)]]),
        # Hough transform
        'hough': True,
        'rho': 1,                   # 1
        'theta': 1*np.pi / 180,     # 1*np.pi/180
        'threshold': 150,           # 50
        'min_line_len': 200,        # 200
        'max_line_gap': 5           # 5
    }


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    if lines is not None:
        draw_lines(line_img, lines)
    return line_img


def weighted_img(img, initial_img, alpha_=0.8, beta_=1., lambda_=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha_, img, beta_, lambda_)


def process_image(image):
    global parameters
    # reading in an image
    modified_image = image.copy()
    # to grayscale
    if parameters['grayscale']:
        modified_image = grayscale(modified_image)
    # gaussian smoothing
    if parameters['smoothing']:
        modified_image = gaussian_blur(modified_image, parameters['kernel_size'])
    # Canny edge detection
    if parameters['canny']:
        modified_image = canny(modified_image, parameters['low_threshold'], parameters['high_threshold'])
    # region of interest
    if parameters['mask']:
        norm_vertices = parameters['norm_vertices'].copy()
        norm_vertices[:, :, 0] *= modified_image.shape[1]
        norm_vertices[:, :, 1] *= modified_image.shape[0]
        vertices = norm_vertices.astype(int)
        modified_image = region_of_interest(modified_image, vertices)
    # Hough transform (returns a 3-D image)
    if parameters['hough']:
        modified_image = hough_lines(modified_image, parameters['rho'], parameters['theta'], parameters['threshold'],
                                     parameters['min_line_len'], parameters['max_line_gap'])
    # output
    output_image = image.copy()
    if parameters['output_filter']:
        output_image = modified_image.copy()
    else:
        # applies filter to input image
        mask = modified_image.copy()
        mask[mask > 0] = 1
        np.place(output_image, mask, 255)
    if parameters['plot_output'] and parameters['debug']:
        plt.imshow(output_image, cmap=parameters['cmap'])
        plt.show()
    return output_image


if __name__ == '__main__':
    if parameters['debug']:
        test_images_paths = ["test_images/" + f for f in os.listdir("test_images/")]
        if parameters['test_image_index'] >= 0:
            test_image_path = test_images_paths[parameters['test_image_index']]
            test_image = mpimg.imread(test_image_path)
            process_image(test_image)
        else:
            for test_image_path in test_images_paths:
                test_image = mpimg.imread(test_image_path)
                process_image(test_image)
    else:
        white_output = 'white.mp4'
        filepath = "challenge.mp4"
        filepath = "solidWhiteRight.mp4"
        clip1 = VideoFileClip(filepath)
        white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
        white_clip.write_videofile(white_output, audio=False)
