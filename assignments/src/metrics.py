import numpy as np

class Metrics:
    def __init__(self, y, y_pred):
        # y and y_pred are 1-d arrays of true values and predicted values
        self.y = y
        self.y_pred = y_pred

    def acc(self):
        # Accuracy
        return 1.0-np.sum(np.abs(np.array(self.y) - np.array(self.y_pred)))/len(self.y)

    def eod(self, s):
        # Equal Opportunity Difference
        # s: list/array of binary values 1/0, sensitive attribute
        # return: EOD = TPR(s=1)-TPR(s=0)
        # TPR = #(y=1, y_pred=1) / #(y=1)
        # Write your code below:
        
        # Convert y and y_pred to numpy arrays for easier manipulation
        y = np.array(self.y)
        y_pred = np.array(self.y_pred)

        # Mask for sensitive attribute s=1
        mask_s1 = np.array(s) == 1
        tpr1 = np.sum((y == 1) & (y_pred == 1) & mask_s1) / np.sum((y == 1) & mask_s1)

        # Mask for sensitive attribute s=0
        mask_s0 = np.array(s) == 0
        tpr0 = np.sum((y == 1) & (y_pred == 1) & mask_s0) / np.sum((y == 1) & mask_s0)

        return tpr1-tpr0

    def aod(self, s):
        # Average Odds Difference
        # s: list/array of binary values 1/0, sensitive attribute
        # return: AOD = (TPR(s=1)-TPR(s=0)+FPR(s=1)-FPR(s=0))/2
        # TPR = #(y=1, y_pred=1) / #(y=1)
        # FPR = #(y=0, y_pred=1) / #(y=0)
        # Write your code below:
        # Convert y and y_pred to numpy arrays for easier manipulation
        y = np.array(self.y)
        y_pred = np.array(self.y_pred)

        # Mask for sensitive attribute s=1
        mask_s1 = np.array(s) == 1
        tpr1 = np.sum((y == 1) & (y_pred == 1) & mask_s1) / np.sum((y == 1) & mask_s1)
        fpr1 = np.sum((y == 0) & (y_pred == 1) & mask_s1) / np.sum((y == 0) & mask_s1)

        # Mask for sensitive attribute s=0
        mask_s0 = np.array(s) == 0
        tpr0 = np.sum((y == 1) & (y_pred == 1) & mask_s0) / np.sum((y == 1) & mask_s0)
        fpr0 = np.sum((y == 0) & (y_pred == 1) & mask_s0) / np.sum((y == 0) & mask_s0)

        return (tpr1 - tpr0 + fpr1 - fpr0)/2.0

    def spd(self, s):
        # Statistical Parity Difference
        # s: list/array of binary values 1/0, sensitive attribute
        # return: SPD = |PR(s=1)-PR(s=0)|
        # PR = #(y_pred=1) / #y
        # Write your code below:
        # Convert y_pred to a numpy array for easier manipulation
        y_pred = np.array(self.y_pred)

        # Mask for sensitive attribute s=1
        mask_s1 = np.array(s) == 1
        pr1 = np.sum(y_pred[mask_s1] == 1) / np.sum(mask_s1)

        # Mask for sensitive attribute s=0
        mask_s0 = np.array(s) == 0
        pr0 = np.sum(y_pred[mask_s0] == 1) / np.sum(mask_s0)

        return pr1 - pr0


