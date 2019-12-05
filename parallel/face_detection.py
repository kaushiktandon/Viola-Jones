import numpy as np
import pickle
from viola_jones import ViolaJones
import time

def train_viola(t, log_file):
    with open("training.pkl", 'rb') as f:
        training = pickle.load(f)
    clf = ViolaJones(T=t)
    # pass size of data set for both test and training
    clf.train(training, 2429, 4548, log_file)
    print("Training evaluation")
    evaluate(clf, training, log_file)
    # Save the model
    clf.save(str(t))

def test_viola(t, log_file):
    with open("test.pkl", 'rb') as f:
        test = pickle.load(f)
    
    clf = ViolaJones.load(str(t))
    print("Testing evaluation")
    evaluate(clf, test, log_file)

def evaluate(clf, data, log_file):
    correct = 0
    all_negatives, all_positives = 0, 0
    true_negatives, false_negatives = 0, 0
    true_positives, false_positives = 0, 0
    classification_time = 0

    for x, y in data:
        if y == 1:
            all_positives += 1
        else:
            all_negatives += 1

        start = time.time()
        prediction = clf.classify(x)
        classification_time += time.time() - start
        if prediction == 1 and y == 0:
            false_positives += 1
        if prediction == 0 and y == 1:
            false_negatives += 1
        
        correct += 1 if prediction == y else 0
    
    log_file_writer = open(log_file, "a+")
    log_file_writer.write("False Positive Rate: %d/%d (%f)\n" % (false_positives, all_negatives, false_positives/all_negatives))
    log_file_writer.write("False Negative Rate: %d/%d (%f)\n" % (false_negatives, all_positives, false_negatives/all_positives))
    log_file_writer.write("Accuracy: %d/%d (%f)\n" % (correct, len(data), correct/len(data)))
    log_file_writer.write("Average Classification Time: %f\n" % (classification_time / len(data)))
    log_file_writer.close()
    
    print("False Positive Rate: %d/%d (%f)" % (false_positives, all_negatives, false_positives/all_negatives))
    print("False Negative Rate: %d/%d (%f)" % (false_negatives, all_positives, false_negatives/all_positives))
    print("Accuracy: %d/%d (%f)" % (correct, len(data), correct/len(data)))
    print("Average Classification Time: %f" % (classification_time / len(data)))

def main():
    log_file = "parallel_output1.txt"
    log_file_writer = open(log_file, "w+")
    log_file_writer.close()

    train_viola(50, log_file)
    test_viola(50, log_file)

if __name__ == '__main__':
    main()