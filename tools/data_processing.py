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
from pathlib import Path
import sys
import faiss
from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics
import torch

from tools.logger import Logger
from tools.plot_tools import plot_confusion_matrix



def evaluate_models(model, model_type, dataset, ref, qry, num_places, n_values, seq_len_list, selected_indices=[], window=2):
    
    num_places_ = len(selected_indices)
    
    data_path = f"outputs/{model}_{model_type}_{dataset}_{ref}_{qry}_{num_places_}/"
    Path(data_path).mkdir(parents=True, exist_ok=True)
    sys.stdout = Logger(data_path, logfile_name=f"logfile")
    tag = f"{model}_{model_type}_{dataset}_{ref}_{qry}_{num_places_}"
    print(f"{tag}")

    distance_matrix, _ = load_distance_matrix_via_feat(model, model_type, dataset, ref, qry, num_places, selected_indices)
        
    VPR_distances_seqslam, recalls = apply_evaluate_seqMatching(data_path, n_values, seq_len_list, distance_matrix, window=window, tag_=tag)
    
    print(f"UPDATE: distance_matrix shape: {distance_matrix.shape}, VPR_distances_seqslam shape: {VPR_distances_seqslam.shape}")
    
    return distance_matrix, VPR_distances_seqslam, recalls


def evaluate_models_org(dataset, ref, qry, num_places, n_values, seq_len_list, selected_indices=[], VPR_models={}, window=2):

    distance_matrix_all = []
    distance_matrix_seqslam_all = []
    recalls_all = []

    for model, model_types in VPR_models.items():
        for model_type in model_types:
            distance_matrix, VPR_distances_seqslam, recalls = evaluate_models(
                model, model_type, dataset, ref, qry, num_places, n_values, seq_len_list, selected_indices, window)
            
            distance_matrix_all.append(distance_matrix)
            distance_matrix_seqslam_all.append(VPR_distances_seqslam)
            recalls_all.append(recalls)

    return np.array(distance_matrix_all), np.array(distance_matrix_seqslam_all), np.array(recalls_all)
    
  
def apply_evaluate_seqMatching(data_path, n_values, seq_len_list, distance_matrix, window=2, tag_="", show_plots=False):
        
    recalls = np.empty((len(seq_len_list), len(n_values)))
    gt = create_custom_gt_indices(ref_size=distance_matrix.shape[0], query_size=distance_matrix.shape[1], window=window)
    
    for seq_len in seq_len_list:
            
        VPR_distances_seqslam = compute_dist_matrix_seqslam(distance_matrix, seq_len=seq_len)
        VPR_prediction_seqslam = np.argsort(VPR_distances_seqslam, axis=0)

        print(f"\nseq len: {seq_len}")
        
        numQ = VPR_distances_seqslam.shape[1]
        recall = compute_recall(gt, np.transpose(VPR_prediction_seqslam), numQ, n_values=n_values, data_path=data_path, name=f"recallAtN_{tag_}_{seq_len}", allow_save=True)
        recalls[seq_len_list.index(seq_len), :] = np.array(list(recall.values()))
            
        if show_plots:
            plot_distance_matrix(VPR_distances_seqslam, gt, data_path=data_path, name=f"DM_{tag_}_{seq_len}_marked", test_results=VPR_prediction_seqslam[0, :], mark=True)

    return VPR_distances_seqslam, recalls


def plot_distance_matrix(distance_matrix, gt, data_path, name, test_results, window=2, mark=False, cmap='Blues'):

    width = 3.5
    golden_ratio = 0.618
    
    import seaborn as sn
    fig = plt.figure(figsize=(width*2, width*golden_ratio*3)) 
    ax = sn.heatmap(distance_matrix, annot=False, cmap=cmap, cbar=True, rasterized=True, square=True)    
    
    if mark: 
        x_values = np.arange(distance_matrix.shape[1])
        y_values = test_results

        if len(x_values) == len(y_values):
            
            for i in range(len(x_values)):
                # if np.isin(y_values[i], gt[i]):         
                if abs(y_values[i] - gt[i]) <= window:
                    plot_formatting = '.g'
                    x = x_values[i] 
                    y = y_values[i] 
                    plt.plot(x, y, plot_formatting, markersize=5)
                
            for i in range(len(x_values)):
                # if np.isin(y_values[i], gt[i]) == False:       
                if abs(y_values[i] - gt[i]) > window:
                    plot_formatting = '.r'
                    x = x_values[i] 
                    y = y_values[i] 
                    plt.plot(x, y, plot_formatting, markersize=5)
    
    xticks = np.arange(0, distance_matrix.shape[1], 50)
    yticks = np.arange(0, distance_matrix.shape[0], 50)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)    
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)    
    ax.set_xlabel("Query")
    ax.set_ylabel("Refarence")
    fig.tight_layout()
    plt.savefig(os.path.join(data_path, f"{name}.png"), bbox_inches='tight')
    plt.close()
    
    
def change_range(features):
    
    features = -0.1 + (features / np.max(features)) * 0.2
    return features 

    
def load_distance_matrix_via_feat(model, model_type, dataset, ref, qry, num_places, selected_indices=[]):
    
    if model == "SAD":
        model_data_path = f"./../SoftwareSuite/outputs/{dataset}_{ref}_{qry}_{num_places}/"
        
        features_ref = np.load(os.path.join(model_data_path, "ref_res.npy"), allow_pickle=True)
        features_qry = np.load(os.path.join(model_data_path, "qry_res.npy"), allow_pickle=True)
        
        features_ref = change_range(features_ref)
        features_qry = change_range(features_qry)
    
    elif model == "NetVLAD": 
        model_data_path = f"./../local_Patch-NetVLAD/patchnetvlad/outputs/NetVLAD_{model_type}_{dataset}_{ref}_{qry}_4096_{num_places}/"
        
        features_ref = np.load(os.path.join(model_data_path, f"../NetVLAD_features/NetVLAD_{model_type}_{dataset}_{ref}_4096_{num_places}.npy"), allow_pickle=True)
        features_qry = np.load(os.path.join(model_data_path, f"../NetVLAD_features/NetVLAD_{model_type}_{dataset}_{qry}_4096_{num_places}.npy"), allow_pickle=True)

    elif model == "Apgem":        
        model_data_path = f"./../deep-image-retrieval/outputs/{model_type}_{dataset}_{ref}_{qry}_{num_places}/"
        
        features_ref = np.load(os.path.join(model_data_path, f"../output_features/features_{model_type}_{dataset}_{ref}_{num_places}.npy"), allow_pickle=True) 
        features_qry = np.load(os.path.join(model_data_path, f"../output_features/features_{model_type}_{dataset}_{qry}_{num_places}.npy"), allow_pickle=True)

    elif model == "MixVPR":        
        model_data_path = f"./../VPR-methods-evaluation/logs/default/mixvpr_{dataset}_{ref}_{qry}_{num_places}/"
        
        features_ref = np.load(os.path.join(model_data_path, f"database_descriptors.npy"), allow_pickle=True)  
        features_qry = np.load(os.path.join(model_data_path, f"queries_descriptors.npy"), allow_pickle=True)
        
    elif model == "CosPlace":        
        model_data_path = f"./../VPR-methods-evaluation/logs/default/cosplace_{dataset}_{ref}_{qry}_{num_places}/"

        features_ref = np.load(os.path.join(model_data_path, f"database_descriptors.npy"), allow_pickle=True)  
        features_qry = np.load(os.path.join(model_data_path, f"queries_descriptors.npy"), allow_pickle=True)
    
    elif model == "EigenPlaces":
        model_data_path = f"./../VPR-methods-evaluation/logs/default/eigenplaces_{dataset}_{ref}_{qry}_{num_places}/"
        
        features_ref = np.load(os.path.join(model_data_path, f"database_descriptors.npy"), allow_pickle=True)
        features_qry = np.load(os.path.join(model_data_path, f"queries_descriptors.npy"), allow_pickle=True)
        
    elif model == "SALAD":
        model_data_path = f"./../VPR-methods-evaluation/logs/default/salad_{dataset}_{ref}_{qry}_{num_places}/"
        
        features_ref = np.load(os.path.join(model_data_path, f"database_descriptors.npy"), allow_pickle=True)
        features_qry = np.load(os.path.join(model_data_path, f"queries_descriptors.npy"), allow_pickle=True)
        
    elif model == "boq":
        model_data_path = f"./../VPR-methods-evaluation/logs/default/boq_{dataset}_{ref}_{qry}_{num_places}/"
        
        features_ref = np.load(os.path.join(model_data_path, f"database_descriptors.npy"), allow_pickle=True)
        features_qry = np.load(os.path.join(model_data_path, f"queries_descriptors.npy"), allow_pickle=True)
        
    
    features_ref_section = features_ref[selected_indices]
    features_qry_section = features_qry[selected_indices]
    
    distance_matrix, predictions = calculate_distance_matrix(features_ref_section, features_qry_section)
    distance_matrix_ = np.transpose(distance_matrix)
    return distance_matrix_, predictions


def calculate_distance_matrix(db_global_descriptors, query_global_descriptors):
    
    pool_size = query_global_descriptors.shape[1]
    faiss_index = faiss.IndexFlatL2(pool_size)
    faiss_index.add(db_global_descriptors)
    distances_org, predictions = faiss_index.search(query_global_descriptors, len(db_global_descriptors))
    distances = reorder_distances(distances_org, predictions)
    return distances, predictions


def reorder_distances(distances, predictions, axis=1):
    
    prediction_indices = np.argsort(predictions, kind='quicksort', axis=axis)
    distances = np.take_along_axis(distances, prediction_indices, axis=axis)
    return distances


def compute_dist_matrix_seqslam(rates_matrix, seq_len=5, use_gpu=False):
    
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    dist_matrix_numpy = torch.from_numpy(rates_matrix)
    dist_matrix = dist_matrix_numpy.to(device).float().unsqueeze(0).unsqueeze(0)
    precomputed_convWeight = torch.eye(seq_len, device=device).unsqueeze(0).unsqueeze(0)
    
    dist_matrix_seqslam = torch.nn.functional.conv2d(dist_matrix, precomputed_convWeight).squeeze()
    dist_matrix_seqslam_np = dist_matrix_seqslam.detach().cpu().numpy()
    return dist_matrix_seqslam_np


def compute_recall(gt, predictions, numQ, n_values, data_path, name, allow_save=True, allow_print=False): 

    correct_at_n = np.zeros(len(n_values))
    for qIx, pred in enumerate(predictions):
        for i, n in enumerate(n_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
    recall_at_n = correct_at_n / numQ
    all_recalls = {}  # make dict for output
    for i, n in enumerate(n_values):
        all_recalls[n] = recall_at_n[i]
        
        if allow_print:
            print("====> Recall {}@{}: {:.4f}".format("Recall", n, recall_at_n[i]))

    if allow_save:
        np.save(data_path + name, all_recalls)
    return all_recalls


def create_custom_gt_indices(ref_size, query_size, window=2):

    # Initialize the GT index matrix with -1 (for any out of bound cases)
    gt_indices = np.full((query_size, 2 * window + 1), -1, dtype=int)

    for j in range(query_size):
        
        start_idx = max(0, j - window)  # Ensure we don't go out of bounds
        end_idx = min(ref_size, j + window + 1)  # Ensure we stay within bounds
        
        valid_indices = np.arange(start_idx, end_idx)
        
        gt_indices[j, :len(valid_indices)] = valid_indices
    return gt_indices


def evaluate_pred(y_test, data_path, y_pred, y_test_names=[], tag="_test"):
    
    accuracy_score = metrics.accuracy_score(y_test, y_pred)
    all_reports = metrics.classification_report(y_test, y_pred, zero_division=0)
    cm = metrics.confusion_matrix(y_test, y_pred)
    
    y_test_names = y_test_names if len(y_test_names) > 0 else np.union1d(y_pred, y_test)
    plot_confusion_matrix(data_path, cm, y_test_names, tag=tag)
        
    print(f"Accuracy score: {accuracy_score:.4f}")
    print(all_reports)
    print(cm)
    
    ratio = diag_offdiag_ratio(cm)
    print(f"Actual train Ratio: {ratio:.4f}")
    
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = metrics.recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = metrics.f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    return cm


def diag_offdiag_ratio(cm):
    if cm.shape == (1, 1):
        return -1

    diag_sum = np.trace(cm)
    off_diag_sum = np.sum(cm) - diag_sum

    if off_diag_sum == 0:
        return float('inf')  # Return infinity if there are no off-diagonal values
    return diag_sum / off_diag_sum


def get_col_distribution(distance_matrix, seq_len, mode="no_SM"):

    n = distance_matrix.shape[1]
    ss = seq_len - 1
    distance_matrix = max_normalise(distance_matrix)
    vertical_slices = []
    
    if mode == "SM":
        for i in range(n):
            vertical_slice = distance_matrix[:, i]
            vertical_slices.append(vertical_slice)
    else:
        for i in range(ss, n):
            vertical_slice = distance_matrix[:, i-ss : i+1]
            vertical_slices.append(vertical_slice)
    
    np.stack(vertical_slices, axis=-1)
    return np.array(vertical_slices)


def max_normalise(array):
    array = np.copy(array)
    max_vals = np.max(array, axis=0)
    
    # Avoid division by zero
    max_vals[max_vals == 0] = 1
    return array / max_vals


def transform_matrices(col_dist_DM, col_dist_DM_SM):
    
    num_samples, num_rows, seq_len = col_dist_DM.shape
    transformed_DM = np.copy(col_dist_DM)
    transformed_DM_SM = np.copy(col_dist_DM_SM)

    for sample_idx in range(num_samples):
        sample = col_dist_DM[sample_idx]
        diagonals = np.flip(np.array([np.diag(sample, k=i) for i in range(-num_rows + 1, 1) if len(np.diag(sample, k=i)) == seq_len]), axis=0)
        
        for diag_idx in range(0, len(diagonals), seq_len):
            diag_block = diagonals[diag_idx:diag_idx + seq_len]
            
            mean_value = np.mean(diag_block)
            for i in range(seq_len):
                np.fill_diagonal(transformed_DM[sample_idx, diag_idx + i:diag_idx + i + seq_len, 0:seq_len], mean_value)
        
        sample_SM = col_dist_DM_SM[sample_idx]
        for row_idx in range(0, len(sample_SM), seq_len):
            row_block = sample_SM[row_idx:row_idx + seq_len]
            
            mean_value = np.mean(row_block)
            transformed_DM_SM[sample_idx, row_idx:row_idx + seq_len] = mean_value

    return transformed_DM, transformed_DM_SM


def transform_matrices_with_window(col_dist_DM, col_dist_DM_SM, window=2):
    
    num_samples, num_rows, seq_len = col_dist_DM.shape
    transformed_DM = np.copy(col_dist_DM)
    transformed_DM_SM = np.copy(col_dist_DM_SM)

    for sample_idx in range(num_samples):
        sample = col_dist_DM[sample_idx]
        diagonals = np.flip(np.array([np.diag(sample, k=i) for i in range(-num_rows + 1, 1) if len(np.diag(sample, k=i)) == seq_len]), axis=0)
        
        for diag_idx in range(len(diagonals)):
            start_idx = max(0, diag_idx - window)
            end_idx = min(len(diagonals), diag_idx + window + 1)
            diag_block = diagonals[start_idx:end_idx]
            
            mean_value = np.mean(diag_block)
            np.fill_diagonal(transformed_DM[sample_idx, diag_idx:diag_idx+seq_len, 0:seq_len], mean_value)
        
        sample_SM = col_dist_DM_SM[sample_idx]
        for row_idx in range(len(sample_SM)):
            start_idx = max(0, row_idx - window)
            end_idx = min(len(sample_SM), row_idx + window + 1)
            row_block = sample_SM[start_idx:end_idx]
            
            mean_value = np.mean(row_block)
            transformed_DM_SM[sample_idx, row_idx] = mean_value

    return transformed_DM, transformed_DM_SM


def get_pr_wrapper(window, dist_matrix, gt, predicted_correct=[]):
    
    mInds = np.argmin(dist_matrix, axis=0) 
    mDists = np.min(dist_matrix, axis=0)
    lb, ub = dist_matrix.min(),dist_matrix.max()
    prvals, tps_ids_all, fps_ids_all, fns_ids_all, tns_ids_all, thresholds = getPRCurve(lb, ub, mInds, mDists, gt, window, predicted_correct=predicted_correct)
    
    return prvals, tps_ids_all, fps_ids_all, fns_ids_all, tns_ids_all, thresholds


def getPRCurve(lb, ub, mInds, mDists, gt, locRad, predicted_correct=[], num_thresholds=2000-1):
    
    prfData = []
    prfData.append([1, 0, 0])
    thresholds = np.linspace(lb, ub, num_thresholds)
    
    tps_ids_all = []
    fps_ids_all = []
    fns_ids_all = []
    tns_ids_all = []
    
    index = 0
    for threshold in thresholds:
        
        within_dist_thresh = mDists <= threshold
        matchFlags = within_dist_thresh if len(predicted_correct) == 0 else (within_dist_thresh & predicted_correct)
        outVals = mInds.copy()
        outVals[~matchFlags] = -1
        
        p,r,f, tp_ids, fp_ids, fn_ids, tn_ids = getPR(outVals,gt,locRad, threshold)
        
        prfData.append([p,r,f])
        tps_ids_all.append(tp_ids)
        fps_ids_all.append(fp_ids)
        fns_ids_all.append(fn_ids)
        tns_ids_all.append(tn_ids)
        index += 1

    return np.array(prfData), tps_ids_all, fps_ids_all, fns_ids_all, tns_ids_all, thresholds


def getPR(mInds,gt,locRad, threshold, show_print=False):
    
    positives = np.argwhere(mInds!=-1)[:,0]
    tp = np.sum(gt[positives] <= locRad)
    fp = len(positives) - tp

    negatives = np.argwhere(mInds==-1)[:,0]
    tn = np.sum(gt[negatives] > locRad)
    fn = len(negatives) - tn

    assert(tp+tn+fp+fn==len(gt))

    if tp == 0:
        return 0,0,0,0,0,0,0 # what else?
    
    tp_ids = positives[np.where(gt[positives] <= locRad)[0]]
    fp_ids = positives[np.where(gt[positives] > locRad)[0]]
    fn_ids = negatives[np.where(gt[negatives] <= locRad)[0]]
    tn_ids = negatives[np.where(gt[negatives] > locRad)[0]]

    prec = tp/float(tp+fp)
    recall = tp/float(tp+fn)
    fscore = 2*prec*recall/(prec+recall)
    
    if show_print:
        print( 'tps: {}, fps:{}, fns:{}, tns:{}, tot:{}, T:{:.2f}, P:{:.2f}, R:{:.2f}'.format(tp, fp, fn, tn, tp+fp+fn+tn, threshold, prec, recall) )
    return prec, recall, fscore, tp_ids, fp_ids, fn_ids, tn_ids


def get_pr_wrapper_pred(window, dist_matrix, gt, data_path, y_decision_proba=[], prf_pred_conf=[], return_confidence=False, tag="VPR", num_thresholds=2000-1):
    
    mInds = np.argmin(dist_matrix, axis=0) 
    mDists = np.min(dist_matrix, axis=0)
    lb, ub = dist_matrix.min(),dist_matrix.max()
    thresholds = np.linspace(lb, ub, num_thresholds)    
    
    if len(prf_pred_conf) != 0:
        
        pred_conf_perct = 1
        prvals, tps_ids_all, fps_ids_all, fns_ids_all, tns_ids_all, probability = getPRCurve_pred_at_conf(mInds, mDists, gt, window, thresholds, y_decision_proba, pred_conf_perct)       
    else: 
        threshold = thresholds[-1] # VPR+SM threshold is the last one (recall = 1)
        prvals, tps_ids_all, fps_ids_all, fns_ids_all, tns_ids_all = getPRCurve_pred_proba(mInds, mDists, gt, window, y_decision_proba, threshold, num_thresholds)
        
        if return_confidence:
            return prvals, tps_ids_all, fps_ids_all, fns_ids_all, tns_ids_all, thresholds

        pred_conf_perct = 1
        prvals, tps_ids_all, fps_ids_all, fns_ids_all, tns_ids_all, probability = getPRCurve_pred_at_conf(mInds, mDists, gt, window, thresholds, y_decision_proba, pred_conf_perct)
    
    return prvals, tps_ids_all, fps_ids_all, fns_ids_all, tns_ids_all, thresholds, probability


def getPRCurve_pred_proba(mInds, mDists, gt, locRad, y_decision_proba=[], threshold=1, num_thresholds=2000-1):
    
    prfData = []
    prfData.append([1, 0, 0])
    
    prediction_class = np.argmax(y_decision_proba, axis=1)
    prediction_proba = np.max(y_decision_proba, axis=1)
    probability_range = np.linspace(np.max(prediction_proba), np.min(prediction_proba), num=num_thresholds)
    
    tps_ids_all = []
    fps_ids_all = []
    fns_ids_all = []
    tns_ids_all = []
    
    print(f"mInds shape: {mInds.shape}, prediction_proba shape: {prediction_proba.shape}, y_decision_proba shape: {y_decision_proba.shape}")
    
    index = 0
    for probability in probability_range:
                
        predicted_wt_confidence = prediction_proba >= probability
        
        within_dist_thresh = mDists <= threshold
        baseline_outVals = mInds.copy()
        baseline_outVals[~within_dist_thresh] = -1
        
        matchFlags = predicted_wt_confidence
        outVals = mInds.copy()
        outVals[~matchFlags] = -1
        
        p,r,f, tp_ids, fp_ids, fn_ids, tn_ids = getPR(outVals,gt,locRad, threshold)   
        prfData.append([p,r,f])
        tps_ids_all.append(tp_ids)
        fps_ids_all.append(fp_ids)
        fns_ids_all.append(fn_ids)
        tns_ids_all.append(tn_ids)
        index += 1
    
    return np.array(prfData), tps_ids_all, fps_ids_all, fns_ids_all, tns_ids_all


def getPRCurve_pred_at_conf(mInds, mDists, gt, locRad, thresholds, y_decision_proba, pred_conf_perct):
    
    prfData = []
    prfData.append([1, 0, 0])
    
    prediction_class = np.argmax(y_decision_proba, axis=1)
    prediction_proba = np.max(y_decision_proba, axis=1)
    
    sorted_proba = np.sort(prediction_proba)
    pred_conf_op = int((1 - pred_conf_perct) * (len(prediction_proba)-1))
    probability = sorted_proba[pred_conf_op]
    
    tps_ids_all = []
    fps_ids_all = []
    fns_ids_all = []
    tns_ids_all = []
    
    index = 0
    for threshold in thresholds:
        
        predicted_wt_confidence = prediction_proba >= probability if pred_conf_perct != 0 else prediction_proba > probability # so that all is included
        predicted_correct = (prediction_class == 0) | (prediction_class == 2)
        
        within_dist_thresh = mDists <= threshold
        baseline_outVals = mInds.copy()
        baseline_outVals[~within_dist_thresh] = -1
        
        matchFlags = []
        for i in range(within_dist_thresh.shape[0]):
            if (predicted_wt_confidence[i]):
                if (within_dist_thresh[i] and predicted_correct[i]):
                    matchFlags.append(True)
                else:
                    matchFlags.append(False)
            else:
                matchFlags.append(within_dist_thresh[i])
        matchFlags = np.array(matchFlags)
        
        outVals = mInds.copy()
        outVals[~matchFlags] = -1
        
        p,r,f, tp_ids, fp_ids, fn_ids, tn_ids = getPR(outVals,gt,locRad, threshold)
        
        prfData.append([p,r,f])
        tps_ids_all.append(tp_ids)
        fps_ids_all.append(fp_ids)
        fns_ids_all.append(fn_ids)
        tns_ids_all.append(tn_ids)
        index += 1

    return np.array(prfData), tps_ids_all, fps_ids_all, fns_ids_all, tns_ids_all, probability


def get_pr_wrapper_pred_wt_replacement(window, dist_matrix, gt, y_decision_proba_all, num_thresholds=2000-1):
    
    mInds = np.argmin(dist_matrix, axis=0) 
    mDists = np.min(dist_matrix, axis=0)
    lb, ub = dist_matrix.min(),dist_matrix.max()
    thresholds = np.linspace(lb, ub, num_thresholds)
    
    pred_conf_perct = 1
    prvals, tps_ids_all, fps_ids_all, fns_ids_all, tns_ids_all = getPRCurve_pred_wt_replacement(mInds, mDists, gt, window, thresholds, y_decision_proba_all, dist_matrix, pred_conf_perct)
    
    return prvals, tps_ids_all, fps_ids_all, fns_ids_all, tns_ids_all, thresholds


def getPRCurve_pred_wt_replacement(mInds, mDists, gt, locRad, thresholds, y_decision_proba_all, dist_matrix, pred_conf_perct, probability2=0, use_max_confidence=True, topN=4):
    
    y_decision_proba = y_decision_proba_all[0, :, :]
    
    prfData = []
    prfData.append([1, 0, 0])

    prediction_class = np.argmax(y_decision_proba, axis=1)
    prediction_proba = np.max(y_decision_proba, axis=1)
    
    sorted_proba = np.sort(prediction_proba)
    pred_conf_op = int((1 - pred_conf_perct) * (len(prediction_proba)-1))
    probability = sorted_proba[pred_conf_op]
    
    tps_ids_all = []
    fps_ids_all = []
    fns_ids_all = []
    tns_ids_all = []
    
    index = 0
    for threshold in thresholds:     
                
        predicted_wt_confidence = prediction_proba >= probability
        predicted_correct = (prediction_class == 0) | (prediction_class == 2)
        
        within_dist_thresh = mDists <= threshold
        baseline_outVals = mInds.copy()
        baseline_outVals[~within_dist_thresh] = -1
        
        all_ranks = np.argsort(dist_matrix, axis=0)
        
        mInds_wt_replc = []
        matchFlags = []
        matchFlags__ = []
        
        for i in range(within_dist_thresh.shape[0]):
            
            if within_dist_thresh[i]:
                
                if predicted_wt_confidence[i] and predicted_correct[i]:  
                    matchFlags.append(True)
                    matchFlags__.append(True)
                    mInds_wt_replc.append(mInds[i])
                    
                else:
                    remaining_confidence_scores = y_decision_proba_all[1:topN, i, :]
                    
                    if use_max_confidence:
                        max_confidence_per_class = np.max(remaining_confidence_scores, axis=0)
                        most_confidence_per_class_rank = np.argmax(remaining_confidence_scores, axis=0) + 1
                        
                        most_confident_score = np.max(max_confidence_per_class)
                        most_confident_rank = np.min(most_confidence_per_class_rank)
                        most_confident_class = np.argmax(max_confidence_per_class)
                        
                        if most_confident_score >= probability2 and most_confident_class in [0, 2]:
                            matchFlags.append(True)                        
                            # replace min index
                            mInds_wt_replc.append(all_ranks[most_confident_rank, i])                        
                        else:
                            matchFlags.append(False)
                            mInds_wt_replc.append(mInds[i])     
                                   
                    matchFlags__.append(False)
            else:
                matchFlags.append(False)
                matchFlags__.append(False)
                mInds_wt_replc.append(mInds[i])
                
        mInds_wt_replc = np.array(mInds_wt_replc)                
        matchFlags = np.array(matchFlags)
        matchFlags__ = np.array(matchFlags__)
        
        outVals__ = mInds.copy()
        outVals__[~matchFlags__] = -1
        
        outVals = mInds_wt_replc.copy()
        outVals[~matchFlags] = -1
        
        p,r,f, tp_ids, fp_ids, fn_ids, tn_ids = getPR(outVals,gt,locRad, threshold)
        
        prfData.append([p,r,f])
        tps_ids_all.append(tp_ids)
        fps_ids_all.append(fp_ids)
        fns_ids_all.append(fn_ids)
        tns_ids_all.append(tn_ids)
        index += 1

    return np.array(prfData), tps_ids_all, fps_ids_all, fns_ids_all, tns_ids_all


