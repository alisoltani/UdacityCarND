#**Finding Lane Lines on the Road** 

### Reflection

###1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipline consists of the following steps:
* First I convert images to grayscale
[image1]: ./examples/grayscale.jpg "Grayscale"
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
