# Viola-Jones
Parallel Implementation of the Viola-Jones algorithm for our EE 451 Course Project

Data Set: http://cbcl.mit.edu/software-datasets/FaceData2.html - Provides set of training and set of test files. Download from www.ai.mit.edu/courses/6.899/lectures/faces.tar.gz

Serial v2 modified from https://github.com/aparande/FaceDetection
```
python3 face_detection.py
```

## Run OpenCV Benchmark
To install OpenCV for python3:
```
pip3 install opencv-python
pip3 install opencv-contrib-python
```

Assuming you are in the `Viola-Jones` directory:
* Create `.txt` file from negative samples
```
sh ./make_neg_info_file.sh
```
* Create `.info` file from positive samples (can do with annotation tool: https://docs.opencv.org/master/dc/d88/tutorial_traincascade.html)
```
sh ./make_pos_info_file.sh
```
* Create `.vec` file from positive samples
```
opencv_createsamples -info face.info -num 2429 -w 24 -h 24 -vec face.vec
```
* Train OpenCV Model
```
opencv_traincascade -data train/classifier -vec face.vec -bg nonface.txt -numPos 2000 -numNeg 4548 -numStages 20 -numThreads 1 -w 24 -h 24 -minhitrate 0.9 -featureType HAAR
```