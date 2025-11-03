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
import matplotlib
import numpy as np
import seaborn as sn
from sklearn.utils import shuffle


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


from tools.data_processing import evaluate_models_org, evaluate_pred, get_col_distribution
from tools.mlp_classifier import extract_features_with_ranks, generate_labels, generate_labels_org, \
    mlp_train, svm_prediction
from tools.dummy_classifier import train_dummy_classifier
from tools.data_processing import create_custom_gt_indices, transform_matrices, transform_matrices_with_window
from tools.logger import Logger


matplotlib.rcParams['ps.fonttype'] = 42
sn.set_context("paper", font_scale=1, rc={"lines.linewidth": 1})

random.seed(0)
np.random.seed(0)



def main(args):
    
    VPR_model_names = ["AP-GeM Res101", "CosPlace", "EigenPlaces", "MixVPR", "NetVLAD P", "SAD", "SALAD"]
    
    VPR_models = {
    "Apgem": ["Resnet101-AP-GeM.pt"],
    "CosPlace": ["SF_XL"],
    "EigenPlaces": ["ResNet50"],
    "MixVPR": ["GCS-Cities"],
    "NetVLAD": ["pittsburgh"], 
    "SAD": ["SAD"],
    "SALAD": ["DINOv2"]
    }
    
    print(f"\n\n\n\n\n\n\n\n\nProcessing seq_len: {args.seq_len}")
    print(f"VPR_model_names: {VPR_model_names}")
    print(f"VPR_models: {VPR_models}")
    
    print(f"\nRunning code with the following parameters:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    
    
    train_length = 500 if args.dataset_name != "SFU-Mountain" else 100
    test_length = 500 if args.dataset_name != "SFU-Mountain" else 100
    num_iter_train = 1
    num_iter_test = 1
    
    topN = 1
    pred_name = "MLP"

    
    n_values = [1, 5, 10, 15, 20, 25]
    seq_len_list = [1, args.seq_len]
    
    output_dir = "outputs" if args.abl_str == "" else f"outputs_ablations{args.abl_str}"
    data_path = f"{output_dir}/classification_{args.dataset_name}_{train_length}_{pred_name}_SL{args.seq_len}{args.abl_str}/"
    Path(data_path).mkdir(parents=True, exist_ok=True)
    sys.stdout = Logger(data_path, logfile_name=f"logfile")
    

    distance_matrix_all_iter = np.empty((num_iter_train, len(VPR_model_names), train_length, train_length))
    new_shape = int(train_length-(args.seq_len-1))
    distance_matrix_seqslam_all_iter = np.empty((num_iter_train, len(VPR_model_names), new_shape, new_shape))
    recalls_all_iter = np.empty((num_iter_train, len(VPR_model_names), len(seq_len_list), len(n_values)))

    # SVM x_train, x_test
    for i in range(num_iter_train):
        
        start_idx = (i * train_length)
        end_idx = ((i+1) * train_length)       
        train_indices = np.arange(start_idx, end_idx)
        
        distance_matrix_all, distance_matrix_seqslam_all, recalls_all = \
            evaluate_models_org(args.dataset_name, args.ref, args.qry, args.num_places, n_values, seq_len_list, train_indices, VPR_models, args.window)
        
        distance_matrix_all_iter[i] = distance_matrix_all
        distance_matrix_seqslam_all_iter[i] = distance_matrix_seqslam_all
        recalls_all_iter[i] = recalls_all

    distance_matrix_all = distance_matrix_all_iter.reshape(-1, *distance_matrix_all_iter.shape[2:])
    distance_matrix_seqslam_all = distance_matrix_seqslam_all_iter.reshape(-1, *distance_matrix_seqslam_all_iter.shape[2:])
    recalls_all = recalls_all_iter.reshape(-1, *recalls_all_iter.shape[2:])

        
    gt = create_custom_gt_indices(ref_size=train_length, query_size=train_length, window=args.window)
    true_indices = np.tile(np.arange(args.seq_len-1, train_length), len(VPR_model_names))
    model_names = np.array([model for model in VPR_model_names for _ in range(args.seq_len-1, train_length)])

        
    # N queries * 9 models
    y_train_org = np.concatenate([generate_labels_org(distance_matrix_before, distance_matrix_after, gt, args.seq_len, args.window)
                for distance_matrix_before, distance_matrix_after in zip(distance_matrix_all, distance_matrix_seqslam_all)])

    y_train_ = np.concatenate(
    [np.stack(generate_labels(distance_matrix_before, distance_matrix_after, gt, args.seq_len, topN=topN))  
     for distance_matrix_before, distance_matrix_after in zip(distance_matrix_all, distance_matrix_seqslam_all)],  
    axis=1  # Concatenate across the second dimension
    )
    y_train = y_train_[0, :]
    print(f"y labels same: {np.array_equal(y_train_org, y_train)}")
    
    vals_, counts_ = np.unique(y_train, return_counts=True)
    print(f"Class distribution before balancing: {dict(zip(vals_, counts_))}\n")
    

    # min max norm applied to entire DM
    col_dist_DM = np.vstack([get_col_distribution(distance_matrix, args.seq_len) for distance_matrix in distance_matrix_all])    
    col_dist_DM_SM = np.vstack([get_col_distribution(distance_matrix, args.seq_len, mode="SM") for distance_matrix in distance_matrix_seqslam_all])
        
    transformed_DM, transformed_DM_SM = transform_matrices(col_dist_DM, col_dist_DM_SM)
    transformed_DM_avg, transformed_DM_SM_avg = transform_matrices_with_window(col_dist_DM, col_dist_DM_SM, window=2)
    
    y_train_all = y_train

    x_train_all_ = np.array([extract_features_with_ranks(col_dist_DM[i], transformed_DM[i], transformed_DM_avg[i], abl_str=args.abl_str, topN=topN) for i in range(col_dist_DM.shape[0])])
    x_train_all = x_train_all_[:, 0, :]
    print(f"x_train_all shape: {x_train_all.shape}")
    
    
    print("\nfor printing feature names")
    feature_names = ["A1: min_sum_rate", "A2: min_value_rate", "A3: global_group_sums_value", "A4: global_block_sums_value"]
    print(feature_names)
            
    vals, counts = np.unique(y_train_all, return_counts=True)    
    print(f"Class distribution before balancing: {dict(zip(vals, counts))}")    
    x_train = x_train_all
    
    print("\nDummy Classifier")
    dummy_clf = train_dummy_classifier(x_train, y_train, data_path)
    y_pred_train_dummy = dummy_clf.predict(x_train) 
    evaluate_pred(y_train, data_path, y_pred_train_dummy, tag="_train_dummy")
    
    print("\nOur predictor")

    x_train, y_train = shuffle(x_train, y_train, random_state=0)
    pipeline, x_train, y_train = mlp_train(x_train, y_train, data_path, pred_name)


    distance_matrix_all_iter = np.empty((num_iter_test, len(VPR_model_names), test_length, test_length))
    new_shape = int(test_length-(args.seq_len-1))
    distance_matrix_seqslam_all_iter = np.empty((num_iter_test, len(VPR_model_names), new_shape, new_shape))
    recalls_all_iter = np.empty((num_iter_test, len(VPR_model_names), len(seq_len_list), len(n_values)))

    
    # SVM x_test, y_test
    for i in range(num_iter_test):
        
        start_idx = (train_length*num_iter_train)
        end_idx = start_idx + test_length      
        test_indices = np.arange(start_idx, end_idx)
        
        distance_matrix_all, distance_matrix_seqslam_all, recalls_all = \
            evaluate_models_org(args.dataset_name, args.ref, args.qry, args.num_places, n_values, seq_len_list, test_indices, VPR_models=VPR_models, window=args.window)
        
        distance_matrix_all_iter[i] = distance_matrix_all
        distance_matrix_seqslam_all_iter[i] = distance_matrix_seqslam_all
        recalls_all_iter[i] = recalls_all

    distance_matrix_all = distance_matrix_all_iter.reshape(-1, *distance_matrix_all_iter.shape[2:])
    distance_matrix_seqslam_all = distance_matrix_seqslam_all_iter.reshape(-1, *distance_matrix_seqslam_all_iter.shape[2:])
    recalls_all = recalls_all_iter.reshape(-1, *recalls_all_iter.shape[2:])
    
    
    gt = create_custom_gt_indices(ref_size=test_length, query_size=test_length, window=args.window)    
    true_indices = np.tile(np.arange(args.seq_len-1, test_length), len(VPR_model_names))
    model_names = np.array([model for model in VPR_model_names for _ in range(args.seq_len-1, test_length)])
    
    y_test_org = np.concatenate([generate_labels_org(distance_matrix_before, distance_matrix_after, gt, args.seq_len, args.window)
                for distance_matrix_before, distance_matrix_after in zip(distance_matrix_all, distance_matrix_seqslam_all)])
    
    y_test_ = np.concatenate(
    [np.stack(generate_labels(distance_matrix_before, distance_matrix_after, gt, args.seq_len, topN=topN))  
    for distance_matrix_before, distance_matrix_after in zip(distance_matrix_all, distance_matrix_seqslam_all)],  
    axis=1  # Concatenate across the second dimension
    )
    y_test = y_test_[0, :]
    print(f"y labels same: {np.array_equal(y_test_org, y_test)}")
    
    col_dist_DM = np.vstack([get_col_distribution(distance_matrix, args.seq_len) for distance_matrix in distance_matrix_all])    
    col_dist_DM_SM = np.vstack([get_col_distribution(distance_matrix, args.seq_len, mode="SM") for distance_matrix in distance_matrix_seqslam_all])
        
    transformed_DM, transformed_DM_SM = transform_matrices(col_dist_DM, col_dist_DM_SM)
    transformed_DM_avg, transformed_DM_SM_avg = transform_matrices_with_window(col_dist_DM, col_dist_DM_SM, window=2)
    
    vals, counts = np.unique(y_test, return_counts=True)    
    print(f"Class distribution all 4 for test: {dict(zip(vals, counts))}")
    

    vals, counts = np.unique(y_test, return_counts=True)    
    print(f"Class distribution for test: {dict(zip(vals, counts))}")

    x_test_ = np.array([extract_features_with_ranks(col_dist_DM[i], transformed_DM[i], transformed_DM_avg[i], abl_str=args.abl_str, topN=topN) for i in range(col_dist_DM.shape[0])])
    x_test = x_test_[:, 0, :]
    
    print("\nDummy Classifier")
    y_pred_test_dummy = dummy_clf.predict(x_test) 
    evaluate_pred(y_test, data_path, y_pred_test_dummy, tag="_val_dummy")
    
    
    print("\nOur predictor")
    y_pred, _ = svm_prediction(pipeline, x_test, y_test)
    evaluate_pred(y_test, data_path, y_pred, tag="_val")
    

    print(f"data_path: {data_path}")
    print("\n\n\n")




if __name__ == "__main__":
    
    
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
    show_plots = True
    
    
    parser = argparse.ArgumentParser(description="Evaluate VPR models with sequence matching.")
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

