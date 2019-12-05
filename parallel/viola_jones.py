"""
A Python implementation of the Viola-Jones ensemble classification method described in 
Viola, Paul, and Michael Jones. "Rapid object detection using a boosted cascade of simple features." Computer Vision and Pattern Recognition, 2001. CVPR 2001. Proceedings of the 2001 IEEE Computer Society Conference on. Vol. 1. IEEE, 2001.
"""
import numpy as np
import math
import pickle
from sklearn.feature_selection import SelectPercentile, f_classif
import time
import threading

class ViolaJones:
    def __init__(self, T = 10):
        """
          Args:
            T: The number of weak classifiers which should be used
        """
        self.T = T
        self.alphas = []
        self.clfs = []

    def update_weights(self, weights, accuracy, beta):
        for i in range(len(accuracy)):
            weights[i] = weights[i] * (beta ** (1 - accuracy[i]))
        return weights

    def log(self, log_file, content):
        log_file_writer = open(log_file, "a+")
        log_file_writer.write(content + "\n")
        log_file_writer.close()

    def train(self, training, pos_num, neg_num, log_file):
        """
        Trains the Viola Jones classifier on a set of images (numpy arrays of shape (m, n))
          Args:
            training: An array of tuples. The first element is the numpy array of shape (m, n) representing the image. The second element is its classification (1 or 0)
            pos_num: the number of positive samples
            neg_num: the number of negative samples
        """
        weights = np.zeros(len(training))
        training_data = []
        self.log(log_file, "Computing integral images")
        print("Computing integral images")

        start_time = time.time()
        for x in range(len(training)):
            training_data.append((integral_image(training[x][0]), training[x][1]))
            if training[x][1] == 1:
                weights[x] = 1.0 / (2 * pos_num)
            else:
                weights[x] = 1.0 / (2 * neg_num)
        end_time = time.time()
        self.log(log_file, (str(end_time - start_time) + " seconds to compute integral images"))
        print (str(end_time - start_time) + " seconds to compute integral images")

        print("Building features")
        start_time = time.time()
        features = self.build_features(training_data[0][0].shape)
        end_time = time.time()
        print (str(end_time - start_time) + " seconds to build features")
        self.log(log_file, (str(end_time - start_time) + " seconds to build features"))

        print("Applying features to training examples")
        start_time = time.time()
        X, y = self.apply_features(features, training_data)
        end_time = time.time()
        print (str(end_time - start_time) + " seconds to apply features")
        self.log(log_file, (str(end_time - start_time) + " seconds to apply features"))

        print("Selecting best features")
        start_time = time.time()
        indices = SelectPercentile(f_classif, percentile=10).fit(X.T, y).get_support(indices=True)
        end_time = time.time()
        print (str(end_time - start_time) + " seconds to select best feature")
        self.log(log_file, (str(end_time - start_time) + " seconds to select best feature"))

        X = X[indices]
        features = features[indices]
        print("Selected %d potential features" % len(X))
        self.log(log_file, "Selected %d potential features" % len(X))

        for t in range(self.T):
            self.log(log_file, "iteration: " + str(t))
            print("iteration: " + str(t))
            start_time = time.time()
            weights = weights / np.linalg.norm(weights)

            start_time = time.time()
            weak_classifiers = self.train_weak(X, y, features, weights)
            end_time = time.time()
            self.log(log_file, str(end_time - start_time) + " seconds to train weak")
            print (str(end_time - start_time) + " seconds to train weak")
            
            start_time = time.time()
            clf, error, accuracy = self.select_best(weak_classifiers, weights, training_data)
            end_time = time.time()
            self.log(log_file, str(end_time - start_time) + " seconds to select best")
            print (str(end_time - start_time) + " seconds to select best")

            start_time = time.time()
            beta = error / (1.0 - error)
            weights = self.update_weights(weights, accuracy, beta)
            end_time = time.time()
            self.log(log_file, str(end_time - start_time) + " seconds to update weights")
            print (str(end_time - start_time) + " seconds to update weights")

            alpha = math.log(1.0/beta)
            self.alphas.append(alpha)
            self.clfs.append(clf)
            print("Chose classifier: %s with accuracy: %f and alpha: %f" % (str(clf), len(accuracy) - sum(accuracy), alpha))
            self.log(log_file, "Chose classifier: %s with accuracy: %f and alpha: %f" % (str(clf), len(accuracy) - sum(accuracy), alpha))

    def threaded_train_weak(self, X, features, weights, y, total_pos, total_neg, thread_id, classifiers, classifiers_lock):
        my_classifiers = list()
        for index, feature in enumerate(X):
            applied_feature = sorted(zip(weights, feature, y), key=lambda x: x[1])

            pos_seen, neg_seen = 0, 0
            pos_weights, neg_weights = 0, 0
            min_error, best_feature, best_threshold, best_polarity = float('inf'), None, None, None
            # Can't really parallelize this because neg_weights/pos_weights in the ith iteration need information from iterations 0 to i - 1
            for w, f, label in applied_feature:
                error = min(neg_weights + total_pos - pos_weights, pos_weights + total_neg - neg_weights)
                if error < min_error:
                    min_error = error
                    best_feature = features[thread_id * 100 + index]
                    best_threshold = f
                    best_polarity = 1 if pos_seen > neg_seen else -1

                if label == 1:
                    pos_seen += 1
                    pos_weights += w
                else:
                    neg_seen += 1
                    neg_weights += w
            
            clf = WeakClassifier(best_feature[0], best_feature[1], best_threshold, best_polarity)
            my_classifiers.append(clf)
        with classifiers_lock:
            for index, clf in enumerate(my_classifiers):
                classifiers[thread_id * 100 + index] = (clf)

    def train_weak(self, X, y, features, weights):
        """
        Finds the optimal thresholds for each weak classifier given the current weights
          Args:
            X: A numpy array of shape (len(features), len(training_data)). Each row represents the value of a single feature for each training example
            y: A numpy array of shape len(training_data). The ith element is the classification of the ith training example
            features: an array of tuples. Each tuple's first element is an array of the rectangle regions which positively contribute to the feature. The second element is an array of rectangle regions negatively contributing to the feature
            weights: A numpy array of shape len(training_data). The ith element is the weight assigned to the ith training example
          Returns:
            An array of weak classifiers
        """
        total_pos, total_neg = 0, 0
        for w, label in zip(weights, y):
            if label == 1:
                total_pos += w
            else:
                total_neg += w

        classifiers = [None] * len(X)
        total_features = X.shape[0]

        # Each iteration of this loop is supposed to train a weak classifier. len(X) is ~5000, so maybe have 1 thread be responsible for 100 classifiers
        num_threads = int(len(X) / 100) + 1
        threads = []
        classifiers_lock = threading.Lock()
        for thread_id in range(num_threads):
            end = min((thread_id + 1) * 100, len(X))
            my_X = X[thread_id * 100 : end]

            my_thread = threading.Thread(target = self.threaded_train_weak, args = (my_X, features, weights, y, total_pos, total_neg, thread_id, classifiers, classifiers_lock))
            threads.append(my_thread)
            my_thread.start()

        for thread in threads:
            thread.join()

        return classifiers
                
    def build_features(self, image_shape):
        """
        Builds the possible features given an image shape
          Args:
            image_shape: a tuple of form (height, width)
          Returns:
            an array of tuples. Each tuple's first element is an array of the rectangle regions which positively contribute to the feature. The second element is an array of rectangle regions negatively contributing to the feature
        """
        height, width = image_shape
        features = []
        for w in range(1, width+1):
            for h in range(1, height+1):
                i = 0
                while i + w < width:
                    j = 0
                    while j + h < height:
                        #2 rectangle features
                        immediate = RectangleRegion(i, j, w, h)
                        right = RectangleRegion(i+w, j, w, h)
                        if i + 2 * w < width: #Horizontally Adjacent
                            features.append(([right], [immediate]))

                        bottom = RectangleRegion(i, j+h, w, h)
                        if j + 2 * h < height: #Vertically Adjacent
                            features.append(([immediate], [bottom]))
                        
                        right_2 = RectangleRegion(i+2*w, j, w, h)
                        #3 rectangle features
                        if i + 3 * w < width: #Horizontally Adjacent
                            features.append(([right], [right_2, immediate]))

                        bottom_2 = RectangleRegion(i, j+2*h, w, h)
                        if j + 3 * h < height: #Vertically Adjacent
                            features.append(([bottom], [bottom_2, immediate]))

                        #4 rectangle features
                        bottom_right = RectangleRegion(i+w, j+h, w, h)
                        if i + 2 * w < width and j + 2 * h < height:
                            features.append(([right, bottom], [immediate, bottom_right]))

                        j += 1
                    i += 1
        return np.array(features)

    def threaded_select_best(self, classifiers, training_data, weights, best_data, thread_id):
        best_clf, best_error, best_accuracy = None, float('inf'), None        
        for clf in classifiers:
            error, accuracy = 0, []
            for data, w in zip(training_data, weights):
                correctness = abs(clf.classify(data[0]) - data[1])
                accuracy.append(correctness)
                error += w * correctness
            error = error / len(training_data)
            if error < best_error:
                best_clf, best_error, best_accuracy = clf, error, accuracy

        best_data[thread_id] = (best_clf, best_error, best_accuracy)

    def select_best(self, classifiers, weights, training_data):
        """
        Selects the best weak classifier for the given weights
          Args:
            classifiers: An array of weak classifiers
            weights: An array of weights corresponding to each training example
            training_data: An array of tuples. The first element is the numpy array of shape (m, n) representing the integral image. The second element is its classification (1 or 0)
          Returns:
            A tuple containing the best classifier, its error, and an array of its accuracy
        """
        # We have ~5000 classifiers. Have threads find the best for their subset and then find the best overall
        num_threads = int(len(classifiers) / 100) + 1
        threads = []
        best_data = [None] * num_threads
        for thread_id in range(num_threads):
            end = min((thread_id + 1) * 100, len(classifiers))
            my_classifiers = classifiers[thread_id * 100 : end]

            my_thread = threading.Thread(target=self.threaded_select_best, args= (my_classifiers, training_data, weights, best_data, thread_id))
            my_thread.start()
            threads.append(my_thread)

        for thread in threads:
            thread.join()

        overall_best_clf, overall_best_error, overall_best_accuracy = None, float('inf'), None
        for data in best_data:
            if data[1] < overall_best_error:
                overall_best_clf = data[0]
                overall_best_error = data[1]
                overall_best_accuracy = data[2]

        return overall_best_clf, overall_best_error, overall_best_accuracy

    def feature_ii_pos(self, training_data, pos_regions, pos_scores):
        for m in range(len(training_data)):
            pos_sum = 0
            ii = training_data[m][0]
            for pos in pos_regions:
                pos_sum += pos.compute_feature(ii)
            pos_scores[m] = pos_sum

    def feature_ii_neg(self, training_data, neg_regions, neg_scores):
        for m in range(len(training_data)):
            neg_sum = 0
            ii = training_data[m][0]
            for neg in neg_regions:
                neg_sum += neg.compute_feature(ii)
            neg_scores[m] = neg_sum

    def feature_ii(self, ii, pos_regions, neg_regions):
        '''
        Helper function for applfying features
        Args:
            ii, pos_regions, neg_regions
        Returns:
            positive weight - negative weight
        '''
        pos_sum = 0
        for pos in pos_regions:
            pos_sum += pos.compute_feature(ii)

        neg_sum = 0
        for neg in neg_regions:
            neg_sum += neg.compute_feature(ii)

        return pos_sum - neg_sum

    def threaded_apply_features_1(self, my_features, training_data, X, X_Lock, thread_id):
        my_thread_output = list()
        for positive_regions, negative_regions in my_features:
            temp_list = list()
            # Could parallelize this too - see threaded_apply_features_2
            for m in range(len(training_data)):
                temp_list.append(self.feature_ii(training_data[m][0], positive_regions, negative_regions))
            my_thread_output.append(temp_list)

        a = 0
        with X_Lock:
            for temp_list in my_thread_output:
                X[thread_id * 1000 + a] = temp_list
                a += 1

    def threaded_apply_features_2(self, my_features, training_data, X, X_Lock, thread_id):
        my_thread_output = list()
        training_data_length = len(training_data)
        for positive_regions, negative_regions in my_features:
            temp_list = [None] * training_data_length
            pos_scores = [None] * training_data_length
            neg_scores = [None] * training_data_length

            pos_thread = threading.Thread(target=self.feature_ii_pos, args=(training_data, positive_regions, pos_scores))
            neg_thread = threading.Thread(target=self.feature_ii_neg, args=(training_data, negative_regions, neg_scores))
            pos_thread.start()
            neg_thread.start()

            pos_thread.join()
            neg_thread.join()

            for i in range(training_data_length):
                temp_list[i] = pos_scores[i] - neg_scores[i]
            my_thread_output.append(temp_list)

        a = 0
        with X_Lock:
            for temp_list in my_thread_output:
                X[thread_id * 1000 + a] = temp_list
                a += 1

    def apply_features(self, features, training_data):
        """
        Maps features onto the training dataset
          Args:
            features: an array of tuples. Each tuple's first element is an array of the rectangle regions which positively contribute to the feature. The second element is an array of rectangle regions negatively contributing to the feature
            training_data: An array of tuples. The first element is the numpy array of shape (m, n) representing the integral image. The second element is its classification (1 or 0)
          Returns:
            X: A numpy array of shape (len(features), len(training_data)). Each row represents the value of a single feature for each training example
            y: A numpy array of shape len(training_data). The ith element is the classification of the ith training example
        """
        X = np.zeros((len(features), len(training_data)))
        y = np.array(list(map(lambda data: data[1], training_data)))
        i = 0
        # With len(features) = 51705, we want 52 threads, 51 responsible for 1000 elements and 1 responsible for ~700 elements
        num_threads = int(len(features) / 1000) + 1
        threads = []
        X_Lock = threading.Lock()
        for thread_id in range(num_threads):
            end = min((thread_id + 1) * 1000, len(features))
            my_features = features[thread_id * 1000 : end]

            # Change to threaded_apply_features_2 to try 2nd version of parallelization
            if thread_id == 0:
                print("Using: threaded_apply_features_1")
            my_thread = threading.Thread(target=self.threaded_apply_features_1, args=(my_features, training_data, X, X_Lock, thread_id))
            threads.append(my_thread)
            my_thread.start()

        for thread in threads:
            thread.join()

        print("Verify X matches!")
        print(X)

        return X, y

    def classify(self, image):
        """
        Classifies an image
          Args:
            image: A numpy 2D array of shape (m, n) representing the image
          Returns:
            1 if the image is positively classified and 0 otherwise
        """
        total = 0
        ii = integral_image(image)
        for alpha, clf in zip(self.alphas, self.clfs):
            total += alpha * clf.classify(ii)
        return 1 if total >= 0.5 * sum(self.alphas) else 0

    def save(self, filename):
        """
        Saves the classifier to a pickle
          Args:
            filename: The name of the file (no file extension necessary)
        """
        with open(filename+".pkl", 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        A static method which loads the classifier from a pickle
          Args:
            filename: The name of the file (no file extension necessary)
        """
        with open(filename+".pkl", 'rb') as f:
            return pickle.load(f)

class WeakClassifier:
    def __init__(self, positive_regions, negative_regions, threshold, polarity):
        """
          Args:
            positive_regions: An array of RectangleRegions which positively contribute to a feature
            negative_regions: An array of RectangleRegions which negatively contribute to a feature
            threshold: The threshold for the weak classifier
            polarity: The polarity of the weak classifier
        """
        self.positive_regions = positive_regions
        self.negative_regions = negative_regions
        self.threshold = threshold
        self.polarity = polarity

    def feature_ii(self, ii):
        pos_sum = 0
        for pos in self.positive_regions:
            pos_sum += pos.compute_feature(ii)

        neg_sum = 0
        for neg in self.negative_regions:
            neg_sum += neg.compute_feature(ii)

        return pos_sum - neg_sum
    
    def classify(self, x):
        """
        Classifies an integral image based on a feature f and the classifiers threshold and polarity
          Args:
            x: A 2D numpy array of shape (m, n) representing the integral image
          Returns:
            1 if polarity * feature(x) < polarity * threshold
            0 otherwise
        """
        if self.polarity * self.feature_ii(x) < self.polarity * self.threshold:
            return 1
        else:
            return 0
    
    def __str__(self):
        return "Weak Clf (threshold=%d, polarity=%d, %s, %s" % (self.threshold, self.polarity, str(self.positive_regions), str(self.negative_regions))

class RectangleRegion:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    def compute_feature(self, ii):
        """
        Computes the value of the Rectangle Region given the integral image
        Args:
            integral image : numpy array, shape (m, n)
            x: x coordinate of the upper left corner of the rectangle
            y: y coordinate of the upper left corner of the rectangle
            width: width of the rectangle
            height: height of the rectangle
        """
        return ii[self.y+self.height][self.x+self.width] + ii[self.y][self.x] - (ii[self.y+self.height][self.x]+ii[self.y][self.x+self.width])

    def __str__(self):
        return "(x= %d, y= %d, width= %d, height= %d)" % (self.x, self.y, self.width, self.height)
    def __repr__(self):
        return "RectangleRegion(%d, %d, %d, %d)" % (self.x, self.y, self.width, self.height)
        
def integral_image(image):
    """
    Computes the integral image representation of a picture. The integral image is defined as following:
    1. s(x, y) = s(x, y-1) + i(x, y), s(x, -1) = 0
    2. ii(x, y) = ii(x-1, y) + s(x, y), ii(-1, y) = 0
    Where s(x, y) is a cumulative row-sum, ii(x, y) is the integral image, and i(x, y) is the original image.
    The integral image is the sum of all pixels above and left of the current pixel
      Args:
        image : an numpy array with shape (m, n)
    """
    ii = np.zeros(image.shape)
    s = np.zeros(image.shape)
    for y in range(len(image)):
        for x in range(len(image[y])):
            s[y][x] = s[y-1][x] + image[y][x] if y-1 >= 0 else image[y][x]
            ii[y][x] = ii[y][x-1]+s[y][x] if x-1 >= 0 else s[y][x]
    return ii