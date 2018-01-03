# **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## Things to Note

- I pickled training images but it's too large so I added it to `.gitignore`
- I used vehicle and non-vehicle data to train the model but not [Udacity dataset](https://github.com/udacity/self-driving-car/tree/master/annotations)
- model is pickle in `model.p`

## Usage

In the command line type `python pipeline.py` with the following optional flags

- with flag `-i` or `--image` to process the images in `./test_images`
- with flag `-v` or `--video` to process `./project_video.mp4`
- with flag `-hm` or `--heatmap` to process the images in `./test_images` with heatmap method

[//]: # (Image References)
[image1]: ./output_images/bboxed_test1.jpg
[image2]: ./output_images/bboxed_test4.jpg
[image3]: ./output_images/bboxed_test5.jpg
[image4]: ./output_images/heat_bboxed_test1.jpg
[image5]: ./output_images/heat_bboxed_test4.jpg
[image6]: ./output_images/heat_bboxed_test5.jpg
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contain in the file 'pipeline.py'. It happens in the method `load_training_data()` which employs a few other helper methods.

The code used to extract HOG features is contained in the file `pipeline_helpers.py`. Method is called `extract_features`. Alternatively you can also extract spatial features or histogram features alongside with HOG features.

But before extracting HOG features you must load all the images and labels to the memory. I first extracted and accumulated all the file names for the images, then I load the images according to them. After that I pickled the image matrices so next time I can load the images directly all at once.

The rest is obvious. You feed the `extract_features` along with all the arguments specified.

#### 2. Explain how you settled on your final choice of HOG parameters.

I used `YUV` channel, orient=11, pixel_per_cell=16, cell_per_block=2. I also tried orient=9. Didn't make much difference in terms of validation accuracy.

Actually the forum has a lot of valuable feedback on how you should choose you HOG parameters or what combinations are the most effective. So I didn't spend much time on parameters tunning. The SVM validation accuracy is above 98%. Good enough for me.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

After I loaded my training data, I used `StandardScaler()` to normalize each column of the data. Before I splitted them into training set and validation set with a ratio of 80% to 20%, one more step was to shuffle the data.

Then I trained a Linear SVM model on the training data and vbalidated it. If the validation accuracy is good, my code will pickle the model for future usage.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for sliding window is in the file `pipeline_helpers.py`. Method is called `one_shot_sliding_window` because together with other methods, for each image I want to draw bounding boxes on, I only extract the HOG features once.

When doing sliding window on test images, I called `multi_scale_sliding_window()` in `pipeline.py`, which utilizes `one_shot_sliding_window()`.

You can pass multiple scales and window size and search area associated with them to `multi_scale_sliding_window()`. Different scales correspond to different window scales. Because we trained SVM on a fixed number of features, which means a fixed number of pixels. Large window means more pixels which won't work with our classifier. So we need to shrink the image and use the fixed sized window to scan it, enlarge the window afterwards.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I scaled each image to 3 different sizes, effectively achieved 3 different sized windows. For the smallest window size I decided to slide it on a smaller region of the image than the other 2 sizes. Because cars close to the camera are too big for this small window size to be useful. It can help to reduce false-positive without losing true-positives.

**Hard-Negative Mining**
I explored the HNM technique, which is great for removing false-positives. The problem is I don't have the data to do so.

I referred to these two posts
1. [HOG object detection](https://www.pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/)
2. [What is Hard-Negative Mining](https://www.reddit.com/r/computervision/comments/2ggc5l/what_is_hard_negative_mining_and_how_is_it/)

When you have images with interested objects in it and they take 10% of the space of the image instead of 90%. Also you know the coordinates of the bounding boxes of the objects. In that case you are good to go with HNM. In our case we don't have that kind of dataset. The best chance is the [Udacity dataset](https://github.com/udacity/self-driving-car/tree/master/annotations), and it was reported erroneous in [this thread](https://discussions.udacity.com/t/udacity-dataset/395818/14)

The method I employed is a simple one. I used `decision_function` that Linear SVC has to tell the confidence score of a classification. In general, true positive has a higher score than a false-positive. I used a threshold to rule out some positives with lower confidence score.

Doing so will always risk losing some true-positives. And the best threshold to take is different for different dataset.

Here is 3 of the resulting images. You can check out `./test_images` for more

![alt text][image1]
![alt text][image2]
![alt text][image3]

Heat maps can be applied to images, too. Here are some output images I produced. Overlapped bounding boxes and false-positives are resolved largely.

![alt text][image4]
![alt text][image5]
![alt text][image6]

I also tried **Non-Maximum-Suppression** from [this blog post](https://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/). It classifies overlapped bounding boxes to groups and use one of them to represent each group. This effectively eliminates the overlapped bounding boxes. But it can't co-exist with heat maps method. I added it to `pipeline_helpers.py` but didn't use it.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./bboxed_project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions, shaking off the false-positives.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Tuning hyperparameters is the major part where my effort went into. The tradeoff here is either you end up with some false-positives in the resulting video and jittering bounding boxes, or you have smoother boxes but lose many true-positives for some frames.

I used a `Box()` instance defined in `pipeline.py` to keep track of the heatmaps of the last n frames. I used weighted sum of the average heatmap over the last n frames and the heatmap of the current frame, as the heatmap for the current frame. The final bounding box coordinates are based on this weighted sum. The hyperparameters include:

- number of frames to track
- heatmap threshold for current heatmap
- heatmap threshold for overall heatmap
- weights for average heatmap and current heatmap

I applied heatmap method to current heatmap before I mixed them to shake off some false-positives beforehand. So I have two thresholds I can tune with. Still it's very difficult to strike a balance. The resulted video is the best I can get.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I made some simple mistakes at the beginning of the project. One is the NaN values. It turned out to be the case that I used `YUV` which produced negative values can caused the problem when using `skimage.feature.hog` with `transform_sqrt=True`. The other is writing image loading methods in a wrong way, trying to eyeball the pattern of file names. Using `glob.glob()` solved it.

About failing, this implementation is not very satisfying. Jittering, missing true-positives are noticable.

About improvement, one quick way is to combine **Hard-Negative Mining** and **Non-Maximum-Suppression** to replace heat maps, if I have the data. And the state-of-the-art are R-CNN and YOLO. I know they have both the accuracy and the performance. I definitely want to try both of them.
