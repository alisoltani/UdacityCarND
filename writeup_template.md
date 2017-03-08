#**Finding Lane Lines on the Road** 

import math

def Grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    
def Canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def GaussianBlur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def RegionOfInterest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def DrawLaneLines(img, lines, color=[255, 0, 0], thickness=2):
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
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def HoughImage(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    DrawLaneLines(line_img, lines)
    return line_img

def HoughLines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an the hough lines.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines


def WeightedImg(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def itertools_flatten(iter_lst):
    return list(itertools.chain(*iter_lst))

def extrapolate_line(line,image, y_1, y_2): # using the mean of the slope, draw a line inside the mask
    
    
    if line:
        line_mean = np.mean(line,0)
        line_mean = itertools_flatten(line_mean)

        m = slope(line_mean)
        b = line_mean[3] - (m * line_mean[2])

        x_1 = int((y_1 - b)/m)
        x_2 = int((y_2 - b)/m)

        line_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

        cv2.line(line_image, (x_1, y_1), (x_2, y_2), [255, 0, 0], 12)
    else: 
        #pdb.set_trace()
        line_image = image
    return line_image

def apply_mask(image, mask_y_top, mask_y_bottom):
    # Apply mask to an image simple fixed mask used, could be improved upon settings shouldnt be called from two different places
    mask_x_left = 450
    mask_x_right = 530
        
    imshape = image.shape
    vertices = np.array([[(mask_x_left,mask_y_top),(mask_x_right, mask_y_top), (0, imshape[0]), (imshape[1],imshape[0])]], dtype=np.int32)
    masked_image = RegionOfInterest(image, vertices)
    
    return masked_image

def slope (line):
    x1, y1, x2, y2 = line
    return ((y2-y1)/(x2-x1))


def process_image(image):
    # settings
    kernel_size = 3                  # for Gaussian blurring
    low_threshold = 60               # for Canny
    high_threshold = 130             # for Canny
    mask_y_top = 320                 # for image mask
    mask_y_bottom = image.shape[1]   # for image mask
    rho = 1                          # distance resolution in pixels of the Hough grid
    theta = np.pi/180                # angular resolution in radians of the Hough grid
    threshold = 25                   # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 25             # minimum number of pixels making up a line
    max_line_gap = 20                # maximum gap in pixels between connectable line segments
    
    empty_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    
    gray = Grayscale(image)

    #Gaussian blurring
    blur_gray = GaussianBlur(gray,kernel_size)

    #Canny settings

    edges = Canny(blur_gray, low_threshold, high_threshold)

    #Display the edges
    #plt.imshow(edges, cmap='Greys_r')


    masked_edges = apply_mask(edges, mask_y_top, mask_y_bottom)


    #Define the Hough transform parameters
    #Make a blank the same size as our image to draw on


    #Run Hough on edge detected image
    hough_image = HoughImage(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    hough_lines = HoughLines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    
    #plt.imshow(hough_image)

    #weighted_image = WeightedImg(hough_image, image, α=0.8, β=1, λ=0.)

    #plt.imshow(weighted_image)

    #DrawLaneLines(line_image, hough_lines, color=[255, 0, 0], thickness=2)

    #plt.imshow(line_image)

    #new_weighted_image = WeightedImg(line_image, image, α=0.8, β=1, λ=0.)

    #plt.imshow(new_weighted_image)

    #if we miss a lane marking, use previous lane marking and start counting, if count goes up too much
    #the functionilaty cant be used.
    
    
    
    #seperate left and right lines by slope
    left_lines  = []
    right_lines = []

    for line in hough_lines:
        for x1,y1,x2,y2 in line:
            if ( slope([x1,y1,x2,y2]) < -0.3 ): # 0.3 is used to get rid of unwanted horizontal lines
                left_lines.append(line)
            elif ( slope([x1,y1,x2,y2]) > 0.3 ):
                right_lines.append(line)
                
    global cache_left_lines
    global cache_right_lines
    
    if left_lines:
        cache_left_lines = left_lines
    else:
        left_lines = cache_left_lines
    if right_lines:
        cache_right_lines = right_lines
    else:
        right_lines = cache_right_lines
    
                
    right_line_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    left_line_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    #DrawLaneLines(right_line_image, right_lines)
    #DrawLaneLines(left_line_image, left_lines)

    #kernel = np.ones((10,10),np.uint8)
    #closing = cv2.morphologyEx(left_line_image, cv2.MORPH_CLOSE, kernel)

    #plt.imshow(left_line_image)

    left_lane_image = extrapolate_line(left_lines,image, mask_y_top, mask_y_bottom)
    right_lane_image = extrapolate_line(right_lines,image, mask_y_top, mask_y_bottom)

    weighted_lanes = WeightedImg(right_lane_image, left_lane_image, α=1, β=1, λ=0.)

    final_weighted_image = WeightedImg(weighted_lanes, image, α=0.8, β=1, λ=0.)
    
    return final_weighted_image
    
import os
os.listdir("test_images/")

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import itertools
import pdb
%matplotlib inline

#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

imageList = os.listdir('test_images/')
for image in imageList:
    final_image = process_image( mpimg.imread('test_images/' + image ) )
    plt.imshow( final_image )
    plt.savefig( 'output/' + image )
    
    

### Reflection

###1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipline consists of the following steps:
* First I convert images to grayscale
* Then apply Gaussian blurring on the image to reduce noise in the image. I use a kernal size of 3.
* The next step I detect the edges in the image using Canny, with a low threshold of 60 and a high of 130.
* A mask is applied to the detected edges. Currently I am using a fixed mask with vertices at [450,320], [530, 320], and the two edges of the image
    * This function can be improved upon to dynamically determine the mask instead of the current fixed solution
* From the output of the masked edges, hough lines are found.
* Once we have the lines, I seperate them into left and right lines based on the slope.
    * In my current solution, when the non-solid line is too small, the algorithm ends up declaring no lines exist. This requires some additional work, either by further tuning parameters in Canny, the mask or Hough, or to develop and algorithm that tracks the lines over frames. Currently I am storing the previous working non-empty line, and using that in such instances. This is something I am currently working on to improve.
* Once the left and right lines are seperated, each line is extrapolated using the slope at the mean point of the left/right lines, and the top of the mask as the intercept point. The lines are drawn down to the bottom of the image.
    * This functionality can be improved upon, and should not be so heavily dependent on the mean.
* Finally, the images are superimposed onto the main image.

![alt text][image1]


###2. Identify potential shortcomings with your current pipeline

Current pipeline is too heavily dependent on settings, which may work in some conditions and not others. For example, if the there are additional lines inside the mask (for example in the the white lane video there was a slightly horizontal line), the averaging of the lines will be heavily affected by this. Old lane lines that are not completly cleaned away also will break the algorithm.
The mask should not be fixed as it is now, or if it is the lane detection algorithm should not depend on it (like it does currently). There should be a way to dynamically track the horizon, or make the camera independent of placement in the car (tilt/position).
Curves are not handled with the simple linear extrapolation that is done, a better fitting has to be done to be able to handle curves.



###3. Suggest possible improvements to your pipeline

A possible improvement would be to make the mask dynamically updated, so it knows where the horizon is.
Another improvement is to change the way that the extrapolation is done, to instead of using the mask points, to use existing points and just stop drawing after the mask. This will help fix slight slope errors. Another way is to use a least squares fit, by finding the line that generates the minimum distance to all the points instead of drawing a single line based on the slope at the mean (this is what I am working on now).
Another improveent for curves is to be able to detect them using a polynomial fit(?).
A potential improvement would be to find a way for the algorithm to update settings so that it can dynamicaly improve the performance.
Finally, improving the way the algorithm uses previous frames in videos is definitely needed.