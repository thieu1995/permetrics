#!/usr/bin/env python
# Created by "Thieu" at 06:18, 26/07/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np


class LabelEncoder:
    def __init__(self):
        self.classes_ = None
        self.encoded_classes_ = None

    def fit(self, y):
        self.classes_, indices = np.unique(y, return_inverse=True)
        self.encoded_classes_ = np.arange(len(self.classes_))
        return self

    def transform(self, y):
        return np.searchsorted(self.classes_, y)

    def inverse_transform(self, y):
        return self.classes_[y]

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)
