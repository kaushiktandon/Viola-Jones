import numpy as np
import features as ft
import sys

class AdaBoost():
    def __init__(self, image_height, image_width):
        self.features = []

        for feature in ft.FEATURE_TYPES.ALL:
            for width in range(feature.dimen[0], image_width, feature.dimen[0]):
                for height in range(feature.dimen[1], image_height, feature.dimen[1]):
                    for x in range(image_width - width):
                        for y in range(image_height - height):
                            self.features.append(
                                feature((x, y), width, height, 0, 1))

    def Learn(self, faces, non_faces, T):
        pos_weight = 1.0 / (2 * len(faces))
        neg_weight = 1.0 / (2 * len(non_faces))
        for face in faces:
            face.SetWeight(pos_weight)
        for non_face in non_faces:
            non_face.SetWeight(neg_weight)

        images = np.hstack((faces, non_faces))

        classifications = dict()
        i = 0
        for feature in self.features:
            # calculate score for each image
            feature_classifications = np.array(([[im, feature.GetClassification(im)] for im in images]))
            classifications[feature] = feature_classifications
            i = i + 1
            if i % 2000 == 0:
                break 

        classifiers = []
        chosen = []

        print ('Selecting classifiers')
        for i in range(T):

            classification_errors = dict()

            # normalize weights
            norm_factor = 1.0 / sum(image.weight for image in images)
            for image in images:
                image.SetWeight(image.weight * norm_factor)

            # select best weak classifier
            for feature, feature_classifications in classifications.items():

                if feature in chosen:
                    continue

                # calculate error
                error = sum(map(lambda im, classification: im.weight if im.label != classification else 0, feature_classifications[:,0], feature_classifications[:,1]))
                # map error -> feature, use error as key to select feature with
                # smallest error later
                classification_errors[error] = feature

            # get best feature (smallest error)
            errors = classification_errors.keys()
            best_error = errors[np.argmin(errors)]
            feature = classification_errors[best_error]
            chosen.append(feature)
            # Don't want to divide by 0 accidentally
            if (best_error == 0):
                best_error = 0.01
            feature_weight = 0.5 * np.log((1-best_error)/best_error)

            classifiers.append((feature, feature_weight))

            # update image weights
            best_feature_classifications = classifications[feature]
            for feature_classification in best_feature_classifications:
                im = feature_classification[0]
                classification = feature_classification[1]
                if im.label != classification:
                    im.SetWeight(im.weight * np.sqrt((1-best_error)/best_error))
                else:
                    im.SetWeight(im.weight * np.sqrt(best_error/(1-best_error)))

        self.classifiers = classifiers

    def TestOnImage(self, image):
        # Compute total score for this image
        total = 0
        for classifier in self.classifiers:
            total += classifier[0].GetClassification(image) * classifier[1]

        if total > 0:
            return 1
        else:
            return -1