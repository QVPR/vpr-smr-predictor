'''
MIT License

Copyright (c) 2025 Somayeh Hussaini, Tobias Fischer and Michael Milford

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
import os
import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin




def train_dummy_classifier(x_train, y_train, data_path):
    
    dummy_clf = ProbabilitySumClassifier()
    dummy_clf.fit(x_train, y_train)
    joblib.dump(dummy_clf, os.path.join(data_path, f"dummy_clf.pkl"))
    
    return dummy_clf




class ProbabilitySumClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.class_probs_ = None

    def fit(self, X, y):
        # Calculate the probability of each class based on y
        unique_classes, counts = np.unique(y, return_counts=True)
        self.class_probs_ = counts / len(y)
        self.classes_ = unique_classes
        return self

    def predict(self, X):
        # Predict based on the class probabilities
        # Assign the class with the highest probability
        # predictions = np.random.choice(self.classes_, size=len(X), p=self.class_probs_)
        predictions = np.zeros((len(X))) # always predict the majority class, class 0
        return predictions
    
    
