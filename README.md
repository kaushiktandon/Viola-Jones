# Viola-Jones
Parallel Implementation of the Viola-Jones algorithm for our EE 451 Course Project

Training Data Set: http://cbcl.mit.edu/software-datasets/FaceData2.html

## Run OpenCV Benchmark
Assuming you are in the `Viola-Jones` directory:
* Create `.txt` file from negative samples
```
sh ./make_neg_info_file.sh
```
* Create `.info` file from positive samples
```
sh ./make_pos_info_file.sh
```
  * How to use annotation tool: https://docs.opencv.org/master/dc/d88/tutorial_traincascade.html 

* Create `.vec` file from positive samples
```
opencv_createsamples -info face.info -num 2429 -w 24 -h 24 -vec face.vec
```
* Train OpenCV Model
```
opencv_traincascade -data train/classifier -vec face.vec -bg nonface.txt -numPos 10 -numNeg 10 -numStages 2 -w 24 -h 24 -minhitrate 0.5 -featureType HAAR
```