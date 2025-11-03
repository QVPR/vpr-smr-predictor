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
from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics
import seaborn as sn
from matplotlib.collections import LineCollection

sn.set_context("paper", font_scale=1, rc={"lines.linewidth": 1})


def plot_PR_curves(data_path, tag, prvals_VPR, prvals_VPR_SM, prvals_VPR_SM_pred_at_confidence, prvals_wt_replc=None):
    
    s = 0.5
    mm = '-'
    plt.figure(figsize=(5, 5*0.618))
    plt.plot(prvals_VPR[:, 1], prvals_VPR[:, 0], mm, color="darkgray", label="VPR", markersize=s)
    plt.plot(prvals_VPR_SM[:, 1], prvals_VPR_SM[:, 0], mm, color="tab:red", label="VPR+SM", markersize=s)
    plt.plot(prvals_VPR_SM_pred_at_confidence[:, 1], prvals_VPR_SM_pred_at_confidence[:, 0], mm, color="tab:blue", label="VPR+SM+Pred (1st filter)", markersize=s)
    if prvals_wt_replc is not None:
        plt.plot(prvals_wt_replc[:, 1], prvals_wt_replc[:, 0], mm, color="tab:green", label="VPR+SM+Pred wt replacement (2nd filter)", markersize=s)

    plt.xticks(np.arange(0, 1.01, 0.25))
    plt.yticks(np.arange(0, 1.01, 0.25))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.legend(loc="best", fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(data_path, f"{tag}.png"), bbox_inches='tight', dpi=200)
    plt.close()
 

def plot_confusion_matrix(data_path, cm, display_labels, tag=""):

    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_percentage = np.nan_to_num(cm_percentage)  # Handle division by zero if there are no samples for a class

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap='Blues', ax=ax)

    for i in range(len(cm)):
        for j in range(len(cm[i])):
            ax.text(j, i, f'\n\n\n({cm_percentage[i, j]*100:.1f}%)', 
                    ha='center', va='center', color='black')

    plt.savefig(os.path.join(data_path, f"confusion_matrix{tag}.png"), bbox_inches='tight', dpi=200)
    plt.close()
    

def plot_pr_curve_with_color_segments(prvals_VPR_SM_pred, data_path, name="pr_pred"):
    
    fig, ax = plt.subplots()

    points = np.array([prvals_VPR_SM_pred[:, 1], prvals_VPR_SM_pred[:, 0]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    lc = LineCollection(segments, cmap='viridis')
    lc.set_array(np.linspace(0, 2, len(prvals_VPR_SM_pred[:, 1])))
    lc.set_linewidth(2)
    ax.add_collection(lc)
    ax.scatter(prvals_VPR_SM_pred[:, 1], prvals_VPR_SM_pred[:, 0], color='black', s=10)
    
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(data_path, f"{name}.png"), bbox_inches='tight', dpi=200)
    plt.close()
    

def plot_mlp_loss(data_path, pipeline):
    
    plt.figure()
    plt.plot(pipeline.named_steps['mlp'].loss_curve_, label='train loss')
    plt.plot(pipeline.named_steps['mlp'].validation_scores_, label='validation score')
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("MLP Training Loss Curve")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(data_path, "mlp_loss_curve_train.png"), bbox_inches='tight', dpi=200)
    plt.close()

