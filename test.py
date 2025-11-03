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
import argparse
import os
from pathlib import Path
import random
import sys
import joblib
import matplotlib
import numpy as np
import seaborn as sn
from sklearn.metrics import auc

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


from tools.mlp_classifier import extract_features_with_ranks, generate_labels, svm_prediction  
from tools.data_processing import evaluate_models, create_custom_gt_indices, transform_matrices, transform_matrices_with_window, \
    evaluate_pred, get_col_distribution, get_pr_wrapper, get_pr_wrapper_pred, get_pr_wrapper_pred_wt_replacement
from tools.plot_tools import plot_PR_curves, plot_pr_curve_with_color_segments
from tools.logger import Logger


matplotlib.rcParams['ps.fonttype'] = 42
sn.set_context("paper", font_scale=1, rc={"lines.linewidth": 1})

random.seed(0)
np.random.seed(0)




def main(args):
        
    print(f"\n\n\n\nProcessing seq_len: {args.seq_len}")

    print(f"\nTesting dataset: {args.dataset_name} | Reference: {args.ref} | Query: {args.qry} | Places: {args.num_places} | Model: {args.model_name} | Model name: {args.model_type}")

    process_data(args.dataset_name, args.ref, args.qry, args.num_places, args.model_name, args.model_type, args.seq_len, args.abl_str)
                

def process_data(dataset_name, ref, qry, num_places, VPR_model_name, VPR_model_type, seq_len, abl_str=""):    
    
    if dataset_name == "nordland":
        train_length = 500
        test_length = 500
        decision_length = 9000
    elif dataset_name == "ORC":
        train_length = 500
        test_length = 500
        decision_length = 1000
    elif dataset_name == "SFU-Mountain":    
        train_length = 100
        test_length = 100
        decision_length = 185
    
    train_length = 500 if dataset_name != "SFU-Mountain" else 100
    test_length = 500 if dataset_name != "SFU-Mountain" else 100
    num_iter_train = 1
    num_iter_test = 1
    
    topN = 4
    window = 2
    epsilon = 1e-6
    
    
    n_values = [1, 5, 10, 15, 20, 25]
    seq_len_list = [1, seq_len]

    output_dir = "outputs" if args.abl_str == "" else f"outputs_ablations{args.abl_str}"
    data_path = f"{output_dir}/classification_{dataset_name}_{train_length}_MLP_SL{seq_len}{abl_str}/"
    Path(data_path).mkdir(parents=True, exist_ok=True)
    sys.stdout = Logger(data_path, logfile_name=f"logfile_applied_{dataset_name}_{VPR_model_name}_{VPR_model_type}.txt")
    
    
    # SVM decision   
    start_idx = (train_length*num_iter_train) + (test_length*num_iter_test)
    end_idx = start_idx + decision_length      
    test_indices = np.arange(start_idx, end_idx)
    
    
    distance_matrix_all, distance_matrix_seqslam_all, recalls_all = evaluate_models(VPR_model_name, VPR_model_type, \
        dataset_name, ref, qry, num_places, n_values, seq_len_list, selected_indices=test_indices, window=window)
    
    print(f"base recall: {recalls_all[0,0]:.4f}")

    gt = create_custom_gt_indices(ref_size=decision_length, query_size=decision_length, window=window)
    true_indices = np.arange(seq_len-1, decision_length)

    y_decision_all = generate_labels(distance_matrix_all, distance_matrix_seqslam_all, gt, seq_len, topN=topN)
    y_decision = y_decision_all[0, :]
    
    vals, counts = np.unique(y_decision, return_counts=True)
    print(f"Class distribution all 4 for decision: {dict(zip(vals, counts))}")    
    
    col_dist_DM = get_col_distribution(distance_matrix_all, seq_len)
    col_dist_DM_SM = get_col_distribution(distance_matrix_all, seq_len, mode="SM")
    transformed_DM, transformed_DM_SM = transform_matrices(col_dist_DM, col_dist_DM_SM)
    transformed_DM_avg, transformed_DM_SM_avg = transform_matrices_with_window(col_dist_DM, col_dist_DM_SM, window=2)
    
    x_decision = np.array([extract_features_with_ranks(col_dist_DM[i], transformed_DM[i], transformed_DM_avg[i], abl_str=abl_str, topN=topN) for i in range(col_dist_DM.shape[0])])

    print("\nPredictions for all queries")
    pipeline_name = f"MLP_pipeline.pkl"
    
    pipeline = joblib.load(os.path.join(data_path, pipeline_name))
    
    y_decision_pred_all = []
    y_decision_proba_all = []
    
    for i in range(x_decision.shape[1]):    
        y_decision_pred, y_decision_proba = svm_prediction(pipeline, x_decision[:, i, :], y_decision_all[i, :])
        y_decision_pred_all.append(y_decision_pred)
        y_decision_proba_all.append(y_decision_proba)
    
    y_decision_pred_all = np.array(y_decision_pred_all)
    y_decision_proba_all = np.array(y_decision_proba_all)
    y_decision_pred = y_decision_pred_all[0, :]
    y_decision_proba = y_decision_proba_all[0, :, :]        
    print(f"shape of y decision: {y_decision.shape}, shape of y_decision_pred: {y_decision_pred.shape}")
    evaluate_pred(y_decision, data_path, y_decision_pred, tag=f"_decision_{VPR_model_name}_{VPR_model_type}")
    
    
    print(f"y decision proba shape: {y_decision_proba.shape}, y decision proba all shape: {y_decision_proba_all.shape}")
    print(f"distance matrix all shape: {distance_matrix_all.shape}, distance matrix seqslam all shape: {distance_matrix_seqslam_all.shape}")
    
    # Scenario: apply SM to all, and use predictor to identify which matches are likely to be incorrect
    print("\nScenario: apply SM to all, and use predictor to identify which matches are likely to be incorrect")
    
    print("\nDummy classifier")
    
    y_decision_rand = np.array([0] * np.sum(y_decision_pred == 0) + [1] * np.sum(y_decision_pred == 1) + \
                            [2] * np.sum(y_decision_pred == 2) + [3] * np.sum(y_decision_pred == 3))    
    
    np.random.shuffle(y_decision_rand)
    np.random.shuffle(y_decision_rand)
    evaluate_pred(y_decision, data_path, y_decision_rand, tag="_decision_rand")
    
    dummy_clf = joblib.load(os.path.join(data_path, "dummy_clf.pkl"))
    y_decision_dummy = dummy_clf.predict(x_decision[:, 0, :])
    evaluate_pred(y_decision, data_path, y_decision_dummy, tag="_decision_dummy")
    
    
    
    print(f"Pred: {np.sum(y_decision_pred == 0)} {np.sum(y_decision_pred == 1)} {np.sum(y_decision_pred == 2)} {np.sum(y_decision_pred == 3)}")
    print(f"Rand: {np.sum(y_decision_rand == 0)} {np.sum(y_decision_rand == 1)} {np.sum(y_decision_rand == 2)} {np.sum(y_decision_rand == 3)}")
    print(f"Dumm: {np.sum(y_decision_dummy == 0)} {np.sum(y_decision_dummy == 1)} {np.sum(y_decision_dummy == 2)} {np.sum(y_decision_dummy == 3)}")
     
    a = np.intersect1d(np.where(y_decision_pred == 2)[0], np.where(y_decision == 2)[0]).shape[0]
    b = np.intersect1d(np.where(y_decision_pred == 1)[0], np.where(y_decision == 2)[0]).shape[0] 
    perct_incorrect_detected = a / (a + b + epsilon)
     
    a_rand = np.intersect1d(np.where(y_decision_rand == 2)[0], np.where(y_decision == 2)[0]).shape[0]
    b_rand = np.intersect1d(np.where(y_decision_rand == 1)[0], np.where(y_decision == 2)[0]).shape[0] 
    perct_incorrect_detected_rand = a_rand / (a_rand + b_rand + epsilon)
    print(f"\npercentage of incorrect matches detected via our predictor: {perct_incorrect_detected:.4f}, via random: {perct_incorrect_detected_rand:.4f}")
    
    
    gt = np.abs(np.arange(0, decision_length) - (np.argmin(distance_matrix_all, axis=0)))
    tag = f"VPR_{dataset_name}_{VPR_model_name}_{VPR_model_type}"
    prvals_VPR, _, _, _, _, _ = get_pr_wrapper(window, distance_matrix_all, gt)
    
    
    gt = np.abs(np.arange(seq_len-1, decision_length) - (np.argmin(distance_matrix_seqslam_all, axis=0)+(seq_len-1)))
    tag = f"VPR_SM_{dataset_name}_{VPR_model_name}_{VPR_model_type}"
    prvals_VPR_SM, _, _, _, _, _ = get_pr_wrapper(window, distance_matrix_seqslam_all, gt)


    gt = np.abs(np.arange(seq_len-1, decision_length) - (np.argmin(distance_matrix_seqslam_all, axis=0)+(seq_len-1)))
    tag = f"pred_conf_{dataset_name}_{VPR_model_name}_{VPR_model_type}"
    prvals_VPR_SM_pred_conf, _, _, _, _, _ = get_pr_wrapper_pred(window, distance_matrix_seqslam_all, gt, data_path, y_decision_proba, return_confidence=True, prf_pred_conf=[], tag=tag)
    plot_pr_curve_with_color_segments(prvals_VPR_SM_pred_conf, data_path, name=f"pr_{tag}_{decision_length}")
    
    
    gt = np.abs(np.arange(seq_len-1, decision_length) - (np.argmin(distance_matrix_seqslam_all, axis=0)+(seq_len-1)))
    tag = f"VPR_SM_pred_at_confidence_{dataset_name}_{VPR_model_name}_{VPR_model_type}"
    prvals_VPR_SM_pred_at_confidence, _, _, _, _, _, _ = get_pr_wrapper_pred(window, distance_matrix_seqslam_all, gt, data_path, y_decision_proba, prf_pred_conf=prvals_VPR_SM_pred_conf, tag=tag)

    
    pr_auc_VPR_SM_pred = auc(prvals_VPR_SM_pred_at_confidence[:, 1], prvals_VPR_SM_pred_at_confidence[:, 0])
    max_recall = prvals_VPR_SM_pred_at_confidence[-1, 1]
    mask = prvals_VPR_SM[:, 1] <= max_recall
    pr_auc_VPR_SM = auc(prvals_VPR_SM[mask, 1], prvals_VPR_SM[mask, 0])
    delta_missed_pr_auc = ((max_recall - pr_auc_VPR_SM) - (max_recall - pr_auc_VPR_SM_pred)) / (max_recall - pr_auc_VPR_SM) * 100
    delta_missed_pr_auc = ((pr_auc_VPR_SM_pred - pr_auc_VPR_SM) / pr_auc_VPR_SM) * 100
    
    print(f"PR AUC VPR+SM: {pr_auc_VPR_SM:.4f}, PR AUC VPR+SM+Pred: {pr_auc_VPR_SM_pred:.4f}, max recall: {max_recall:.4f}")
    print(f"Percentage reduction in PR Area Over the Curve vs VPR+SM: {delta_missed_pr_auc:.2f}%")

    tag = f"VPR_SM_Pred_{dataset_name}_{VPR_model_name}_{VPR_model_type}"
    plot_PR_curves(data_path, tag, prvals_VPR, prvals_VPR_SM, prvals_VPR_SM_pred_at_confidence)
    

    gt = np.abs(np.arange(seq_len-1, decision_length) - (np.argmin(distance_matrix_seqslam_all, axis=0)+(seq_len-1)))
    prvals_wt_replc, _, _, _, _, _ = get_pr_wrapper_pred_wt_replacement(window, distance_matrix_seqslam_all, gt, y_decision_proba_all)
    
    tag = f"VPR_SM_Pred_wt_replacement_{dataset_name}_{VPR_model_name}_{VPR_model_type}"
    plot_PR_curves(data_path, tag, prvals_VPR, prvals_VPR_SM, prvals_VPR_SM_pred_at_confidence, prvals_wt_replc=prvals_wt_replc)




if __name__ == '__main__': 
    
    model_name = "NetVLAD"
    model_type = "pittsburgh"
    
    # dataset_name = "nordland"
    # ref = "summer"
    # qry = "winter"
    # num_places = 27592
    # seq_len = 4
    # window = 2
    # show_plots = False
    
    # dataset_name = "ORC"
    # ref = "Rain"
    # qry = "Dusk"
    # num_places = 3800
    # seq_len = 4
    # window = 2
    # show_plots = False
    
    dataset_name = "SFU-Mountain"
    ref = "dry"
    qry = "dusk"
    num_places = 385
    seq_len = 4
    window = 2
    show_plots = False
    

    parser = argparse.ArgumentParser(description="Evaluate VPR models with sequence matching.")
    parser.add_argument("--model_name", type=str, default=model_name, help="VPR model name.")
    parser.add_argument("--model_type", type=str, default=model_type, help="VPR model type.")
    parser.add_argument("--dataset_name", type=str, default=dataset_name, help="Dataset name (e.g., 'nordland').")
    parser.add_argument("--ref", type=str, default=ref, help="Reference set (e.g., 'summer').")
    parser.add_argument("--qry", type=str, default=qry, help="Query set (e.g., 'winter').")
    parser.add_argument("--num_places", type=int, default=num_places, help="Number of places to consider (default: 100).")
    parser.add_argument("--seq_len", type=int, default=seq_len, help="Sequence length for matching (e.g., 4).")
    parser.add_argument("--window", type=int, default=window, help="Window size for sequence matching (default: 2).")
    parser.add_argument("--abl_str", type=str, default="", help="Ablation string for output directory (default: '').")
    parser.add_argument("--show_plots", action='store_true', default=show_plots, help="Whether to show plots during evaluation (default: False).")
    
    args = parser.parse_args()

    main(args)
    
