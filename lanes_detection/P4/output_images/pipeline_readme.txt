Images for every stage of the advanced lanes detection pipeline:

*_undistorted —-> undistorted version of the input image
*_binary      --> sobelx and s channel thresholding
*_warped      --> birds-eye view's perspective transform
*_lane_pixels --> sliding window search
*_poly        --> quadratic fit 
*_final       —-> final version of the processed image 