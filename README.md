[box_matching]: ./images/0003_boxmatch.png "box_matching"
[ttc_lidar_01]: ./images/ttc_lidar_01.png "ttc_lidar_01"
[ttc_lidar_02]: ./images/ttc_lidar_02.png "ttc_lidar_02"
[kpt_box]: ./images/kpt_box.png "kpt_box"
[ttc-camera_formula_01]: ./images/ttc-camera_formula_01.png "ttc-camera_formula_01"
[ttc-camera_formula_02]: ./images/ttc-camera_formula_02.png "ttc-camera_formula_02"
[ttc_lidar_006]: ./images/ttc_lidar_006.png "ttc_lidar_006"
[ttc_lidar_007]: ./images/ttc_lidar_007.png "ttc_lidar_007"
[ttc_lidar_008]: ./images/ttc_lidar_008.png "ttc_lidar_008"
[ttc_camera_compare]: ./images/ttc_camera_compare.png "ttc_camera_compare"
[ttc_compare]: ./images/ttc_compare.png "ttc_compare"
[HARRIS]: ./images/HARRIS.png "HARRIS"
[SHITOMASI]: ./images/SHITOMASI.png "SHITOMASI"
[BRISK]: ./images/BRISK.png "BRISK"
[FAST]: ./images/FAST.png "FAST"
[ORB]: ./images/ORB.png "ORB"
[SIFT]: ./images/SIFT.png "SIFT"


# SFND 3D Object Tracking

Welcome to the final project of the camera course. By completing all the lessons, you now have a solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, you know how to detect objects in an image using the YOLO deep-learning framework. And finally, you know how to associate regions in a camera image with Lidar points in 3D space. Let's take a look at our program schematic to see what we already have accomplished and what's still missing.

<img src="images/course_code_structure.png" width="779" height="414" />

In this final project, you will implement the missing parts in the schematic. To do this, you will complete four major tasks: 
1. First, you will develop a way to match 3D objects over time by using keypoint correspondences. 
2. Second, you will compute the TTC based on Lidar measurements. 
3. You will then proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. And lastly, you will conduct various tests with the framework. Your goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. In the last course of this Nanodegree, you will learn about the Kalman filter, which is a great way to combine the two independent TTC measurements into an improved version which is much more reliable than a single sensor alone can be. But before we think about such things, let us focus on your final project in the camera course. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.

---

## [Rubric](https://review.udacity.com/#!/rubrics/2550/view) Points

### FP.1 Match 3D Objects
Below image is the screenshot of box matching result. Box can be matched correctly from previous frame to current frame.

![box_matching]

### FP.2 Compute Lidar-based TTC
Below image shows the ttc-lidar result. This is a normal case, there is no outlier lidar points in top-view image, so the ttc-lidar value is same with ttc-lidar without outlier.

![ttc_lidar_01]

There is an outlier lidar point, it shows in top-view image. You can see that the result of ttc-lidar value is havily effected by this outlier (`ttc_lidar = 3.83 s`), but the result of ttc-lidar without outlier is still stady (`ttc_lidar_without_outlier = 14.98 s`).

![ttc_lidar_02]

### FP.3 Associate Keypoint Correspondences with Bounding Boxes

I use `cv::findHomography` to filter out the outlier keypoint matches. Below image shows the result of keypoint matches which are in the relative bounding box. Blue lines are inlier matches (I just show top 10 best matches in this image). Green lines are outliers which will not be used to compute ttc-camere in next step. You can see that the outlier matches are successfully removed from the current bounding box (A keypoint on the tree in previous frame is mis-matched with the forward car in current frame).

![kpt_box]

### FP.4 Compute Camera-based TTC

According the formula introduced in the Udacity course, I implement the function to calculate ttc-lidar by the keypoints found in previous step (Only using the inlier keypoint matches).

![ttc-camera_formula_01]

![ttc-camera_formula_02]

### FP.5 Performance Evaluation 1
Since the ttc-lidar value is calculated according to the formula `TTC = minXCurr * delta_time / (minXPrev - minXCurr);`. I think it will be huge affected by the difference between previous and current frame's minimum x. In the below example, the value of `minXPrev - minXCurr` in frame 007 is significiently small, so the ttc-lidar is suddently increase at this time and decrease in next frame.

**Frame 006: `minXPrev - minXCurr = 0.0609999`**
![ttc_lidar_006]

**Frame 007: `minXPrev - minXCurr = 0.0220003`**
![ttc_lidar_007]

**Frame 008: `minXPrev - minXCurr = 0.0799999`**
![ttc_lidar_008]

### FP.6 Performance Evaluation 2
Below are the testing result of each detector and descriptor. Using HARRIS or ORB as detector basiclly can't work, too mutch keypoint matches are outlier. SIFT+ORB will run out of memory.

![HARRIS]

![SHITOMASI]

![BRISK]

![FAST]

![ORB]

![SIFT]

I choose several combinations from above detector/descriptor which have good performance to compare together. According to the spreadsheet and line graph, I think the top 2 stable ttc-camera are using *SIFT+SIFT* and *AKAZE+AKAZE* as detector and descriptor. Their result are most close to the ttc-lidar at the same frame, except frame_7 and frame_8. I think the gap are coming from lidar's outlier at that two frame. But considering the conclusion of mid-term project, I will choose *FAST+BRIEF* since it seems only has one ttc-camera result which is way off, and it consumes less time than *SIFT+SIFT* and *AKAZE+AKAZE*. If we can improve our method of filtering the outlier from keypoint match, I think *FAST+BRIEF* can be more stable. So my top 3 choose are:
1. FAST+BRIEF
2. AKAZE+AKAZE
3. SIFT+SIFT
   
![ttc_compare]

![ttc_camera_compare]