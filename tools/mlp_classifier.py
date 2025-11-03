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
import warnings
import joblib
import numpy as np
from sklearn import model_selection
from sklearn.discriminant_analysis import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from tools.data_processing import evaluate_pred
from tools.plot_tools import plot_mlp_loss




def svm_prediction(pipeline, x_test, y_test):
    
    vals, counts = np.unique(y_test, return_counts=True)
    print(f"Class distribution (Test): {dict(zip(vals, counts))}")
    
    y_pred = pipeline.predict(x_test)
    y_pred_proba = pipeline.predict_proba(x_test)
    
    return y_pred, y_pred_proba


def mlp_train(x_train, y_train, data_path, pred_name):
    
    # Check class distribution to determine appropriate k_neighbors for SMOTE
    vals, counts = np.unique(y_train, return_counts=True)
    min_samples = min(counts)
    print(f"Class distribution before SMOTE: {dict(zip(vals, counts))}")
    
    # Set k_neighbors to be at most min_samples - 1 to avoid the error
    # SMOTE needs at least 2 samples per class, so we ensure k_neighbors >= 1
    k_neighbors = max(1, min(5, min_samples - 1))
    print(f"Using k_neighbors={k_neighbors} for SMOTE (min class samples: {min_samples})")
    
    smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=k_neighbors)
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)
    print(f"Class distribution after SMOTE: {dict(zip(*np.unique(y_train_resampled, return_counts=True)))}")
    
    scaler_x = StandardScaler()
    mlp = MLPClassifier(hidden_layer_sizes=(128, 128, 128), activation='relu', solver='adam', early_stopping=True, 
                        max_iter=10000, random_state=0) # batch_size=64, 

    pipeline = Pipeline([
        ('scaler', scaler_x),
        ('mlp', mlp)
    ])

    skf = model_selection.StratifiedKFold(n_splits=5)
    cross_val_scores = model_selection.cross_val_score(pipeline, x_train_resampled, y_train_resampled, 
                                                       cv=skf, scoring='f1_macro')
    print(f"Cross-validation F1 Macro Score (mean over 5 folds): {cross_val_scores.mean():.4f}")

    param_grid = {
        'mlp__hidden_layer_sizes': [(128, 128, 128)],  
        'mlp__activation': ['relu'],
        'mlp__solver': ['adam'],
        'mlp__alpha': [0.0001],  # 0.001
        'mlp__learning_rate_init': [0.001],
        'mlp__max_iter': [10000]
    }
    
    grid_search = model_selection.GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_macro', refit=True)
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
        grid_search.fit(x_train_resampled, y_train_resampled)

    pipeline = grid_search.best_estimator_
    y_train_pred = pipeline.predict(x_train_resampled)
    print(f"Best parameters found: {grid_search.best_params_}")

    evaluate_pred(y_train_resampled, data_path, y_train_pred, tag="_train")

    joblib.dump(pipeline, data_path + f"{pred_name}_pipeline.pkl")

    plot_mlp_loss(data_path, pipeline)
    print(f"Number of iterations used: {pipeline.named_steps['mlp'].n_iter_}")

    return pipeline, x_train_resampled, y_train_resampled


def extract_features_with_ranks(sample, sample_block, sample_group, abl_str="", topN=4, epsilon=1e-6):

    _, num_cols = sample.shape
    seq_len = num_cols

    # Compute all top-left to bottom-right diagonals
    diagonals = get_diagonals(sample)    
    diagonals_block = get_diagonals(sample_block)
    diagonals_group = get_diagonals(sample_group)
    diagonal_sums = np.array([np.sum(diag) for diag in diagonals])

    # Get indices of the top 4 minimum diagonal sums
    top_indices = np.argsort(diagonal_sums)[:topN]  # Get top1, top2, top3, top4 indices

    row_means = np.mean(sample[:,:-1], axis=1)
    low_row_mean_indices = np.argsort(row_means)[:topN]
        
    diagonal_means = np.mean(diagonals[:,:-1], axis=1)
    low_diag_mean_indices = np.argsort(diagonal_means)[:topN]

    features = np.zeros((topN, 4))

    for rank, idx in enumerate(top_indices):  # rank 0 to 3 (for 4 values)
        
        min_value = np.sort(sample[:, -1])[rank]
        min_sum_min_value = diagonals[idx, -1]
        min_value_rate = min_value / (min_sum_min_value + epsilon)
        
        min_row_sum = np.sum(sample[low_row_mean_indices[rank], :-1])
        min_diag_sum = np.sum(diagonals[low_diag_mean_indices[rank], :-1])
        min_sum_rate = min_diag_sum / (min_row_sum + epsilon)
    
        global_min_value = np.sum(diagonals[idx, :-1])    
        global_block_sum_value = np.sum(diagonals_block[idx])
        global_group_sum_value = np.sum(diagonals_group[idx])
        
        global_group_sums_value = global_min_value / (global_group_sum_value + epsilon)
        global_block_sums_value = global_min_value / (global_block_sum_value + epsilon)

        # Store features in structured array
        if abl_str == "_A1":
            features[rank] = [min_sum_rate]
        elif abl_str == "_A2":
            features[rank] = [min_value_rate]
        elif abl_str == "_A3":
            features[rank] = [global_group_sums_value]
        elif abl_str == "_A4":
            features[rank] = [global_block_sums_value]
        elif abl_str == "":
            features[rank] = [min_sum_rate, min_value_rate, global_group_sums_value, global_block_sums_value]

    return features


def get_diagonals(sample):
    
    num_rows, num_cols = sample.shape
    diagonals = np.flip(np.array([np.diag(sample, k=i) for i in range(-num_rows + 1, 1) if len(np.diag(sample, k=i)) == num_cols]), axis=0)
    
    return diagonals


def generate_labels(distance_matrix_before, distance_matrix_after, gt_matrix, seq_len, topN=4):
    """
    Generates numeric labels based on whether the prediction was correct or incorrect 
    before and after applying sequence matching.

    Parameters:
    - distance_matrix_before: Distance matrix before sequence matching (100x100, query x reference)
    - distance_matrix_after: Distance matrix after sequence matching (97x97, query x reference)
    - gt_matrix: Ground truth matrix (100 x list of 10 correct indices)
    - seq_len: Sequence length for matching (causes the offset between before and after matrices)
    - window: The rank window for a match to be considered correct (default=2)
    
    Returns:
    - y_labels: List of numeric labels for each query in distance_matrix_after
    """
    
    y_labels_topN = [[] for _ in range(topN)]  # Store separate y_labels lists (one per top rank)
    num_queries = distance_matrix_after.shape[1]
    
    for i in range(num_queries):
        # Adjust indices to align with the sequence length shift
        adjusted_index = i + (seq_len - 1)

        # Get the sorted ranks (lower is better) before and after sequence matching
        ranks_before = np.argsort(distance_matrix_before[:, adjusted_index])  # Before sequence matching
        ranks_after = np.argsort(distance_matrix_after[:, i]) + (seq_len - 1) # After sequence matching

        # Select the topN ranked matches before and after sequence matching
        topN_before = ranks_before[:topN]  # Top N closest before matching
        topN_after = ranks_after[:topN]    # Top N closest after matching

        # Compute y_labels for each of the top N ranks
        for rank_idx in range(topN):
            top_before = topN_before[rank_idx]
            top_after = topN_after[rank_idx]

            # Check correctness using ground truth indices
            correct_before = any((np.abs(top_before - gt) == 0) for gt in gt_matrix[adjusted_index] if gt != -1)
            correct_after = any((np.abs(top_after - gt) == 0) for gt in gt_matrix[adjusted_index] if gt != -1)

            # Assign y_labels based on correctness before and after sequence matching
            if correct_before and correct_after:
                y_labels_topN[rank_idx].append(0)  # Correct before & correct after
            elif correct_before and not correct_after:
                y_labels_topN[rank_idx].append(1)  # Correct before & incorrect after
            elif not correct_before and correct_after:
                y_labels_topN[rank_idx].append(2)  # Incorrect before & correct after
            else:
                y_labels_topN[rank_idx].append(3)  # Incorrect before & incorrect after

    return np.array([np.array(y_labels) for y_labels in y_labels_topN])


def generate_labels_org(distance_matrix_before, distance_matrix_after, gt_matrix, seq_len, window=2):
    """
    Generates numeric labels based on whether the prediction was correct or incorrect 
    before and after applying sequence matching.

    Parameters:
    - distance_matrix_before: Distance matrix before sequence matching (100x100, query x reference)
    - distance_matrix_after: Distance matrix after sequence matching (97x97, query x reference)
    - gt_matrix: Ground truth matrix (100 x list of 10 correct indices)
    - seq_len: Sequence length for matching (causes the offset between before and after matrices)
    - window: The rank window for a match to be considered correct (default=2)
    
    Returns:
    - y_labels: List of numeric labels for each query in distance_matrix_after
    """
    
    y_labels = []
    num_queries = distance_matrix_after.shape[1]
    
    for i in range(num_queries):
        # Adjust indices to align with the sequence length shift
        adjusted_index = i + (seq_len - 1)
        
        # Get the sorted ranks (lower is better) before and after sequence matching
        ranks_before = np.argsort(distance_matrix_before[:, adjusted_index])  # Use adjusted index for 'before' matrix
        ranks_after = np.argsort(distance_matrix_after[:, i]) + (seq_len - 1) # 'after' matrix adjusted 

        # Check if any ground truth index is within tolerance (top-k matches) - loop through the gt with tolerance 
        correct_before = any((np.abs(ranks_before[0]-gt) == 0) for gt in gt_matrix[adjusted_index] if gt != -1)
        correct_after = any((np.abs(ranks_after[0]-gt) == 0) for gt in gt_matrix[adjusted_index] if gt != -1)

        if correct_before and correct_after:
            y_labels.append(0)  # "was correct before sequence matching, & now correct after sequence matching"
        elif correct_before and not correct_after:
            y_labels.append(1)  # "was correct before sequence matching, & now incorrect after sequence matching"
        elif not correct_before and correct_after:
            y_labels.append(2)  # "was incorrect before sequence matching, & now correct after sequence matching"
        else:
            y_labels.append(3)  # "was incorrect before sequence matching, & now incorrect after sequence matching"
    
    return np.array(y_labels)

