# Viola-Jones
Hyeon Uk Jeong, Andrew Opem, Kaushik Tandon

Parallel Implementation of the Viola-Jones algorithm for our EE 451 Course Project

Data Set: http://cbcl.mit.edu/software-datasets/FaceData2.html - Provides set of training and set of test files. Download from www.ai.mit.edu/courses/6.899/lectures/faces.tar.gz

To run our parallel version
```
pip3 install -r requirements.txt
cd parallel
python3 face_detection.py
```

## OpenCV Benchmark
### Install
To install OpenCV for python3:
```
pip3 install opencv-python
pip3 install opencv-contrib-python
```
* If OpenCV not detected after above steps:
```
sudo apt install libopencv-dev
```

### Run
Assuming you are in the `Viola-Jones/` directory:
* Create `.txt` file from negative samples
* Create `.info` file from positive samples (can do with annotation tool: https://docs.opencv.org/master/dc/d88/tutorial_traincascade.html)
* Create `.vec` file from positive samples
* Train OpenCV Model: adjust `-numThreads` to number of desired threads

```
bash ./make_neg_info_file.sh
bash ./make_pos_info_file.sh
opencv_createsamples -info face.info -num 2429 -w 24 -h 24 -vec face.vec
mkdir train/classifier
opencv_traincascade -data train/classifier -vec face.vec -bg nonface.txt -numPos 1500 -numNeg 4548 -numStages 20 -numThreads 1 -w 24 -h 24 -minHitRate 0.99 -maxFalseAlarmRate 0.4 -featureType HAAR
```

To visualize OpenCV results
```
python3 cv_benchmark.py
```

Serial version modified from:
* https://github.com/aparande/FaceDetection
* https://github.com/aashudwivedi/object_detect/
* https://github.com/alexdemartos/ViolaAndJones
* https://github.com/paveyry/FaceDetection