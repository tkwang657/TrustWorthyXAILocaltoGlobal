#! /usr/bin/env python3
"""
Local Rule Extraction for TabNet using Influence Functions + Rule Induction

This module extracts faithful, human-readable local decision rules for TabNet 
predictions using Influence Functions. Designed for eventual aggregation into 
global rules
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import torch
import matplotlib.pyplot as plt
from matplotlib import pyplot


class LocalRuleExtractor:
    """
    Extracts local symbolic rules from TabNet predictions using influence functions.
    
    Pipeline:
    1. Identify influential training samples (S_plus, S_minus)
    2. Cluster influential samples by features, logits, and influence sign
    3. Derive local symbolic rules from each cluster using decision trees
    """
    
    def __init__(self, model, feature_names: List[str], n_clusters: int = 3, 
                 max_depth: int = 5, min_samples_split: int = 10,
                 categorical_indices: Optional[List[int]] = None):
        """
        Args:
            model: Trained TabNet model (wrapped with TabNetWrapper)
            feature_names: List of feature names for rule interpretation
            n_clusters: Number of clusters for influential samples
            max_depth: Maximum depth for decision trees
            min_samples_split: Minimum samples to split a node
            categorical_indices: List of indices for categorical features (to preserve categorical nature)
        """
        self.model = model
        self.feature_names = feature_names
        self.n_clusters = n_clusters
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.categorical_indices = categorical_indices if categorical_indices is not None else []
        self.model.eval()
    
    def get_influential_samples_from_csv(self, csv_path: str, test_id: int,
                                         top_k: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Step 1: Identify influential training samples from CSV file.
        
        Args:
            csv_path: Path to CSV file with columns: train_id, test_id, influence
            test_id: Test sample ID
            top_k: Number of top helpful/harmful samples to consider
            
        Returns:
            helpful_indices: Indices of helpful samples in training set
            harmful_indices: Indices of harmful samples in training set
            helpful_influences: Influence values for helpful samples
            harmful_influences: Influence values for harmful samples
        """
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Filter for this test_id
        test_df = df[df['test_id'] == test_id].copy()
        
        if len(test_df) == 0:
            raise ValueError(f"No influences found for test_id {test_id} in CSV")
        
        # Sort by influence and get top helpful (positive) and harmful (negative)
        test_df = test_df.sort_values('influence', ascending=False)
        
        # Get helpful (positive influence) and harmful (negative influence)
        helpful_df = test_df[test_df['influence'] > 0].head(top_k)
        harmful_df = test_df[test_df['influence'] < 0].tail(top_k)
        
        helpful_indices = helpful_df['train_id'].values.astype(int)
        harmful_indices = harmful_df['train_id'].values.astype(int)
        helpful_influences = helpful_df['influence'].values
        harmful_influences = harmful_df['influence'].values
        
        return helpful_indices, harmful_indices, helpful_influences, harmful_influences
    
    def get_influential_samples(self, influence_results: Dict, test_id: int, 
                                top_k: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Step 1: Identify influential training samples.
        
        Args:
            influence_results: Results from calc_influence_dataset
            test_id: Test sample ID
            top_k: Number of top helpful/harmful samples to consider
            
        Returns:
            S_plus_features: Feature values of helpful samples
            S_minus_features: Feature values of harmful samples
            S_plus_indices: Indices of helpful samples in training set
            S_minus_indices: Indices of harmful samples in training set
        """
        if str(test_id) not in influence_results:
            raise ValueError(f"Test ID {test_id} not found in influence_results")
        
        result = influence_results[str(test_id)] 
        influences = np.array(result['influence']) # influence of each training point
        helpful_indices = np.array(result['helpful'][:top_k]) # most helpful training points
        harmful_indices = np.array(result['harmful'][:top_k]) # most harmful training points
        
        return helpful_indices, harmful_indices, influences[helpful_indices], influences[harmful_indices]
    
    def get_model_predictions(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get model predictions and logits for given samples.
        
        Args:
            X: Feature matrix
            
        Returns:
            predictions: Predicted classes
            logits: Raw model outputs (before softmax)
        """
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            logits = self.model(X_tensor).numpy()
            predictions = np.argmax(logits, axis=1)
        
        return predictions, logits
    
    def cluster_influential_samples(self, X_train: np.ndarray, 
                                    helpful_indices: np.ndarray,
                                    harmful_indices: np.ndarray,
                                    helpful_influences: np.ndarray,
                                    harmful_influences: np.ndarray,
                                    clustering_method: str = 'kmeans') -> Dict:
        """
        Step 2: Cluster influential samples by features, logits, and influence sign.
        
        Args:
            X_train: Full training feature matrix
            helpful_indices: Indices of helpful training samples
            harmful_indices: Indices of harmful training samples
            helpful_influences: Influence values for helpful samples
            harmful_influences: Influence values for harmful samples
            clustering_method: 'kmeans' or 'dbscan'
            
        Returns:
            Dictionary with cluster assignments and metadata
        """
        # Get features and predictions for influential samples
        S_plus_features = X_train[helpful_indices]
        S_minus_features = X_train[harmful_indices]
        
        S_plus_preds, S_plus_logits = self.get_model_predictions(S_plus_features)
        S_minus_preds, S_minus_logits = self.get_model_predictions(S_minus_features)
        
        # Combine all influential samples
        all_features = np.vstack([S_plus_features, S_minus_features])
        all_logits = np.vstack([S_plus_logits, S_minus_logits])
        all_influences = np.concatenate([helpful_influences, harmful_influences])
        all_labels = np.concatenate([S_plus_preds, S_minus_preds])
        all_indices = np.concatenate([helpful_indices, harmful_indices])
        influence_signs = np.concatenate([np.ones(len(helpful_indices)), 
                                         -np.ones(len(harmful_indices))])
        
        # Separate categorical and continuous features
        n_samples, n_features = all_features.shape
        categorical_mask = np.array([i in self.categorical_indices for i in range(n_features)])
        continuous_mask = ~categorical_mask
        
        # Handle categorical features: use mode-based clustering or distance metric
        # For categorical features, we'll use a distance-based approach that preserves categorical nature
        categorical_features = all_features[:, categorical_mask] if categorical_mask.any() else None
        continuous_features = all_features[:, continuous_mask] if continuous_mask.any() else None
        
        # Prepare clustering features
        clustering_parts = []
        
        # Continuous features: standardize
        if continuous_features is not None and continuous_features.shape[1] > 0:
            scaler = StandardScaler()
            continuous_scaled = scaler.fit_transform(continuous_features)
            clustering_parts.append(continuous_scaled)
        else:
            scaler = None
        
        # Categorical features: use one-hot encoding or mode-based distance
        # For now, we'll use a simple approach: treat categorical as ordinal but use mode for cluster centers
        if categorical_features is not None and categorical_features.shape[1] > 0:
            # For categorical, we can use a distance metric that works with integers
            # Or use mode-based clustering. For simplicity, we'll scale them differently
            # Use min-max scaling to [0, 1] range to preserve relative distances
            from sklearn.preprocessing import MinMaxScaler
            cat_scaler = MinMaxScaler()
            categorical_scaled = cat_scaler.fit_transform(categorical_features)
            clustering_parts.append(categorical_scaled)
        else:
            cat_scaler = None
        
        # Add logits, influences, and influence signs (these are always continuous)
        logits_scaled = StandardScaler().fit_transform(all_logits)
        influences_scaled = StandardScaler().fit_transform(all_influences.reshape(-1, 1))
        
        clustering_parts.extend([logits_scaled, influences_scaled, influence_signs.reshape(-1, 1)])
        
        # Combine all features for clustering
        clustering_features = np.hstack(clustering_parts) if len(clustering_parts) > 1 else clustering_parts[0]
        
        # Perform clustering
        # For mixed data, we can still use kmeans but with better initialization
        # Alternatively, use a method that handles mixed data better
        if clustering_method == 'kmeans':
            # Use kmeans with k-means++ initialization (works better for mixed data)
            clusterer = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10, init='k-means++')
            cluster_labels = clusterer.fit_predict(clustering_features)
        elif clustering_method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = clusterer.fit_predict(clustering_features)
        elif clustering_method == 'mode':
            # Mode-based clustering for categorical data
            # Cluster based on mode of categorical features + kmeans for continuous
            cluster_labels = self._mode_based_clustering(
                all_features, categorical_mask, self.n_clusters
            )
            clusterer = None  # Mode-based doesn't return a clusterer object
        else:
            raise ValueError(f"Unknown clustering method: {clustering_method}. Use 'kmeans', 'dbscan', or 'mode'")
        
        return {
            'cluster_labels': cluster_labels,
            'features': all_features,
            'logits': all_logits,
            'predictions': all_labels,
            'influences': all_influences,
            'indices': all_indices,
            'influence_signs': influence_signs,
            'scaler': scaler,
            'cat_scaler': cat_scaler,
            'clusterer': clusterer,
            'categorical_mask': categorical_mask
        }
    
    def _mode_based_clustering(self, features: np.ndarray, categorical_mask: np.ndarray, 
                               n_clusters: int) -> np.ndarray:
        """
        Mode-based clustering for mixed categorical/continuous data.
        
        For categorical features, uses mode (most common value) per cluster.
        For continuous features, uses kmeans.
        """
        n_samples = features.shape[0]
        
        # If we have categorical features, use them for initial clustering
        if categorical_mask.any():
            categorical_features = features[:, categorical_mask]
            
            # Group samples by their categorical feature combinations
            # Create a hash for each sample's categorical values
            cat_hashes = []
            for i in range(n_samples):
                cat_hash = hash(tuple(categorical_features[i].astype(int)))
                cat_hashes.append(cat_hash)
            
            # Find unique categorical patterns
            unique_patterns = {}
            for i, cat_hash in enumerate(cat_hashes):
                if cat_hash not in unique_patterns:
                    unique_patterns[cat_hash] = []
                unique_patterns[cat_hash].append(i)
            
            # Use categorical patterns for initial clustering
            cluster_labels = np.zeros(n_samples, dtype=int)
            
            if len(unique_patterns) <= n_clusters:
                # Fewer patterns than clusters: assign each pattern to a cluster
                for cluster_id, (pattern_hash, indices) in enumerate(unique_patterns.items()):
                    for idx in indices:
                        cluster_labels[idx] = cluster_id
                
                # If we have fewer patterns than desired clusters, split larger clusters
                if len(unique_patterns) < n_clusters:
                    # Use continuous features to further split clusters
                    continuous_features = features[:, ~categorical_mask] if (~categorical_mask).any() else None
                    if continuous_features is not None and continuous_features.shape[1] > 0:
                        # Refine clusters using continuous features
                        from sklearn.cluster import KMeans
                        next_cluster_id = len(unique_patterns)
                        for pattern_id in range(len(unique_patterns)):
                            pattern_mask = cluster_labels == pattern_id
                            if pattern_mask.sum() > 1 and next_cluster_id < n_clusters:
                                # Split this cluster using continuous features
                                sub_features = continuous_features[pattern_mask]
                                n_sub_clusters = min(n_clusters - next_cluster_id + 1, len(sub_features), pattern_mask.sum())
                                if n_sub_clusters > 1:
                                    sub_kmeans = KMeans(n_clusters=n_sub_clusters, 
                                                       random_state=42, n_init=10)
                                    sub_labels = sub_kmeans.fit_predict(sub_features)
                                    # Update cluster labels
                                    for idx, sub_label in zip(np.where(pattern_mask)[0], sub_labels):
                                        cluster_labels[idx] = next_cluster_id + sub_label - 1
                                    next_cluster_id += n_sub_clusters
            else:
                # More patterns than clusters: group similar patterns
                # Create pattern representatives (one sample per pattern)
                pattern_reps = []
                pattern_to_idx = {}
                for pattern_idx, (pattern_hash, indices) in enumerate(unique_patterns.items()):
                    # Use first sample of each pattern as representative
                    pattern_reps.append(categorical_features[indices[0]])
                    pattern_to_idx[pattern_hash] = pattern_idx
                
                # Cluster patterns using kmeans
                from sklearn.cluster import KMeans
                pattern_reps_array = np.array(pattern_reps)
                cat_kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                pattern_cluster_map = cat_kmeans.fit_predict(pattern_reps_array)
                
                # Create mapping from pattern hash to cluster
                pattern_hash_to_cluster = {}
                for pattern_hash, pattern_idx in pattern_to_idx.items():
                    pattern_hash_to_cluster[pattern_hash] = pattern_cluster_map[pattern_idx]
                
                # Assign samples to clusters based on their pattern's cluster
                for i, cat_hash in enumerate(cat_hashes):
                    cluster_labels[i] = pattern_hash_to_cluster[cat_hash]
            
            return cluster_labels
        
        # Fallback to kmeans on all features
        from sklearn.cluster import KMeans
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return clusterer.fit_predict(features)
    
    def extract_rules_from_cluster(self, cluster_data: Dict, cluster_id: int,
                                   X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """
        Step 3: Derive local symbolic rules from a cluster using decision trees.
        
        Args:
            cluster_data: Output from cluster_influential_samples
            cluster_id: ID of cluster to extract rules from
            X_train: Full training feature matrix
            y_train: Full training labels
            
        Returns:
            Dictionary with explanation and counterfactual rules
        """
        # Get samples in this cluster
        cluster_mask = cluster_data['cluster_labels'] == cluster_id
        if cluster_mask.sum() < self.min_samples_split:
            return {
                'cluster_id': cluster_id,
                'n_samples': cluster_mask.sum(),
                'explanation_rule': None,
                'counterfactual_rule': None,
                'tree': None
            }
        
        cluster_features = cluster_data['features'][cluster_mask]
        cluster_labels = cluster_data['predictions'][cluster_mask]
        cluster_indices = cluster_data['indices'][cluster_mask]
        cluster_influences = cluster_data['influences'][cluster_mask]
        
        # Fit decision tree on cluster
        tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=42
        )
        tree.fit(cluster_features, cluster_labels)
        
        # Extract explanation rule (most common prediction path)
        most_common_label = np.bincount(cluster_labels).argmax()
        
        # Use comprehensive rule extraction with multiple fallback strategies
        explanation_rule = self._extract_rule_comprehensive(
            tree, cluster_features, cluster_labels, most_common_label
        )
        
        # Find counterfactual: nearest cluster with different dominant label
        counterfactual_rule = None
        try:
            counterfactual_rule = self._find_counterfactual_rule(
                cluster_data, cluster_id, cluster_features, most_common_label,
                X_train, y_train
            )
        except (ValueError, AttributeError):
            pass  # Counterfactual is optional
        
        return {
            'cluster_id': cluster_id,
            'n_samples': cluster_mask.sum(),
            'dominant_label': int(most_common_label),
            'mean_influence': float(cluster_influences.mean()),
            'explanation_rule': explanation_rule,
            'counterfactual_rule': counterfactual_rule,
            'tree': tree,
            'cluster_indices': cluster_indices.tolist(),
            'cluster_features': cluster_features.tolist() if len(cluster_features) < 100 else None  # Store for visualization
        }
    
    def _extract_rule_from_tree(self, tree: DecisionTreeClassifier, 
                                X: np.ndarray, target_label: int,
                                is_explanation: bool = True) -> Optional[str]:
        """Extract a human-readable rule from decision tree.
        
        Returns None if no valid path can be found (instead of vague rule).
        """
        # Find paths leading to target label using sklearn's decision_path
        paths = []
        for i in range(len(X)):
            sample = X[i:i+1]
            pred = tree.predict(sample)[0]
            
            if pred == target_label:
                # Use sklearn's decision_path for accurate path tracing
                path = self._trace_tree_path_sklearn(tree, sample[0], target_label)
                if path and len(path) > 0:
                    paths.append(path)
        
        if not paths:
            # Try alternative: find any leaf node with target_label
            path = self._find_any_path_to_label(tree, target_label)
            if path:
                paths.append(path)
        
        if not paths:
            # If still no paths, return None instead of vague rule
            return None
        
        # Use most common path or simplest path
        if is_explanation:
            # For explanation, use the most common path
            path_str = self._paths_to_rule(paths, target_label)
        else:
            # For counterfactual, use minimal change path
            path_str = self._paths_to_rule(paths, target_label, minimal=True)
        
        return path_str
    
    def _trace_tree_path_sklearn(self, tree: DecisionTreeClassifier, sample: np.ndarray,
                                 target_label: int) -> Optional[List[Tuple[str, float, str]]]:
        """Trace a path through the decision tree using sklearn's decision_path."""
        sample = sample.reshape(1, -1)
        decision_path = tree.decision_path(sample)
        node_indicator = decision_path.toarray()[0]
        
        # Get the leaf node reached
        leaf_id = tree.apply(sample)[0]
        
        # Check if leaf predicts target_label
        leaf_value = tree.tree_.value[leaf_id][0]
        if np.argmax(leaf_value) != target_label:
            return None
        
        # Trace path from root to leaf
        path = []
        node_id = 0
        
        while node_id != leaf_id:
            if node_indicator[node_id] == 0:
                break
                
            # Check if this is a leaf
            if tree.tree_.children_left[node_id] == tree.tree_.children_right[node_id]:
                break
            
            feature_idx = tree.tree_.feature[node_id]
            threshold = tree.tree_.threshold[node_id]
            
            if feature_idx < 0:  # Invalid feature index
                break
                
            feature_name = self.feature_names[feature_idx] if feature_idx < len(self.feature_names) else f"feature_{feature_idx}"
            
            # Determine which child to go to
            if sample[0][feature_idx] <= threshold:
                path.append((feature_name, threshold, "<="))
                node_id = tree.tree_.children_left[node_id]
            else:
                path.append((feature_name, threshold, ">"))
                node_id = tree.tree_.children_right[node_id]
        
        return path if len(path) > 0 else None
    
    def _find_any_path_to_label(self, tree: DecisionTreeClassifier, target_label: int) -> Optional[List[Tuple[str, float, str]]]:
        """Find any path in the tree that leads to target_label by examining leaf nodes."""
        # Find all leaf nodes with target_label
        leaves_with_label = []
        for node_id in range(tree.tree_.node_count):
            if tree.tree_.children_left[node_id] == tree.tree_.children_right[node_id]:  # Leaf node
                leaf_value = tree.tree_.value[node_id][0]
                if np.argmax(leaf_value) == target_label:
                    leaves_with_label.append(node_id)
        
        if not leaves_with_label:
            return None
        
        # Trace path from root to first matching leaf
        target_leaf = leaves_with_label[0]
        path = []
        node_id = 0
        
        def trace_to_leaf(node_id, target_leaf, path):
            if node_id == target_leaf:
                return path, True
            
            if tree.tree_.children_left[node_id] == tree.tree_.children_right[node_id]:
                return path, False  # Reached a leaf but not the target
            
            feature_idx = tree.tree_.feature[node_id]
            threshold = tree.tree_.threshold[node_id]
            
            if feature_idx < 0:
                return path, False
            
            feature_name = self.feature_names[feature_idx] if feature_idx < len(self.feature_names) else f"feature_{feature_idx}"
            
            # Try left child
            left_path = path + [(feature_name, threshold, "<=")]
            result_path, found = trace_to_leaf(tree.tree_.children_left[node_id], target_leaf, left_path)
            if found:
                return result_path, True
            
            # Try right child
            right_path = path + [(feature_name, threshold, ">")]
            result_path, found = trace_to_leaf(tree.tree_.children_right[node_id], target_leaf, right_path)
            if found:
                return result_path, True
            
            return path, False
        
        path, found = trace_to_leaf(node_id, target_leaf, [])
        return path if found and len(path) > 0 else None
    
    def _extract_rule_from_tree_with_labels(self, tree: DecisionTreeClassifier,
                                           X: np.ndarray, y: np.ndarray) -> Optional[str]:
        """Extract rule using actual labels from the cluster (fallback method)."""
        # Get unique labels in cluster
        unique_labels = np.unique(y)
        if len(unique_labels) == 0:
            return None
        
        # Try to extract rule for the most common label
        most_common_label = np.bincount(y).argmax()
        
        # Find samples with this label
        label_mask = y == most_common_label
        if label_mask.sum() == 0:
            return None
        
        # Use first sample with this label
        sample_idx = np.where(label_mask)[0][0]
        sample = X[sample_idx:sample_idx+1]
        
        # Trace path
        path = self._trace_tree_path_sklearn(tree, sample[0], most_common_label)
        if path and len(path) > 0:
            conditions = []
            for feature_name, threshold, op in path:
                if op == "<=":
                    conditions.append(f"{feature_name} <= {threshold:.4f}")
                else:
                    conditions.append(f"{feature_name} > {threshold:.4f}")
            return "IF " + " AND ".join(conditions) + f" THEN class = {most_common_label}"
        
        return None
    
    def _extract_rule_comprehensive(self, tree: DecisionTreeClassifier,
                                   X: np.ndarray, y: np.ndarray, 
                                   target_label: int) -> str:
        """
        Comprehensive rule extraction using multiple strategies.
        Ensures a rule is always returned for each cluster.
        
        Based on techniques from "Local Rule-Based Explanations of Black Box Decision Systems"
        
        Strategy 1: Extract all paths to target label and use most representative
        Strategy 2: Extract rule from tree structure directly (all leaves with target)
        Strategy 3: Use sample-based path extraction
        Strategy 4: Generate rule from most important features in cluster
        """
        # Strategy 1: Extract all paths to target label from tree structure
        # This is the most robust method - directly extracts from tree structure
        all_paths = self._extract_all_paths_to_label(tree, target_label)
        if all_paths:
            # Use the path with the most samples (most representative)
            # This ensures we get the most common pattern in the cluster
            best_path = max(all_paths, key=lambda p: p['n_samples'])
            if best_path['path'] and len(best_path['path']) > 0:
                conditions = []
                for feature_name, threshold, op in best_path['path']:
                    if op == "<=":
                        conditions.append(f"{feature_name} <= {threshold:.4f}")
                    else:
                        conditions.append(f"{feature_name} > {threshold:.4f}")
                return "IF " + " AND ".join(conditions) + f" THEN class = {target_label}"
        
        # Strategy 2: Try sample-based path extraction
        paths = []
        for i in range(len(X)):
            sample = X[i:i+1]
            pred = tree.predict(sample)[0]
            if pred == target_label:
                path = self._trace_tree_path_sklearn(tree, sample[0], target_label)
                if path and len(path) > 0:
                    paths.append(path)
        
        if paths:
            # Use the first path (simplest)
            path = paths[0]
            conditions = []
            for feature_name, threshold, op in path:
                if op == "<=":
                    conditions.append(f"{feature_name} <= {threshold:.4f}")
                else:
                    conditions.append(f"{feature_name} > {threshold:.4f}")
            return "IF " + " AND ".join(conditions) + f" THEN class = {target_label}"
        
        # Strategy 3: Extract rule from any leaf node with target label
        path = self._find_any_path_to_label(tree, target_label)
        if path and len(path) > 0:
            conditions = []
            for feature_name, threshold, op in path:
                if op == "<=":
                    conditions.append(f"{feature_name} <= {threshold:.4f}")
                else:
                    conditions.append(f"{feature_name} > {threshold:.4f}")
            return "IF " + " AND ".join(conditions) + f" THEN class = {target_label}"
        
        # Strategy 4: Generate rule from tree root to first leaf with target label
        # This is a fallback that always works
        rule = self._extract_rule_from_tree_structure(tree, target_label)
        if rule:
            return rule
        
        # Final fallback: Generate a simple rule based on cluster statistics
        return self._generate_statistical_rule(X, y, target_label)
    
    def _extract_all_paths_to_label(self, tree: DecisionTreeClassifier, 
                                    target_label: int) -> List[Dict]:
        """
        Extract all paths from root to leaves that predict target_label.
        Returns list of paths with their sample counts.
        Based on techniques from "Local Rule-Based Explanations of Black Box Decision Systems"
        """
        paths = []
        
        def traverse_tree(node_id, current_path):
            """Recursively traverse tree to find all paths to target label."""
            # Check if this is a leaf node
            if tree.tree_.children_left[node_id] == tree.tree_.children_right[node_id]:
                # Leaf node
                leaf_value = tree.tree_.value[node_id][0]
                predicted_label = np.argmax(leaf_value)
                n_samples = int(leaf_value[predicted_label])
                
                if predicted_label == target_label:
                    paths.append({
                        'path': current_path.copy(),
                        'n_samples': n_samples,
                        'leaf_node': node_id
                    })
                return
            
            # Internal node - continue traversal
            feature_idx = tree.tree_.feature[node_id]
            threshold = tree.tree_.threshold[node_id]
            
            if feature_idx < 0:
                return
            
            feature_name = self.feature_names[feature_idx] if feature_idx < len(self.feature_names) else f"feature_{feature_idx}"
            
            # Left child (<= threshold)
            left_path = current_path + [(feature_name, threshold, "<=")]
            traverse_tree(tree.tree_.children_left[node_id], left_path)
            
            # Right child (> threshold)
            right_path = current_path + [(feature_name, threshold, ">")]
            traverse_tree(tree.tree_.children_right[node_id], right_path)
        
        # Start traversal from root
        traverse_tree(0, [])
        return paths
    
    def _extract_rule_from_tree_structure(self, tree: DecisionTreeClassifier,
                                         target_label: int) -> Optional[str]:
        """
        Extract rule directly from tree structure by finding the shortest path
        to a leaf node that predicts target_label.
        """
        # Find all leaf nodes with target_label
        target_leaves = []
        for node_id in range(tree.tree_.node_count):
            if tree.tree_.children_left[node_id] == tree.tree_.children_right[node_id]:
                leaf_value = tree.tree_.value[node_id][0]
                if np.argmax(leaf_value) == target_label:
                    target_leaves.append(node_id)
        
        if not target_leaves:
            return None
        
        # Find shortest path to first target leaf
        target_leaf = target_leaves[0]
        path = []
        node_id = 0
        
        def find_path_to_leaf(current_node, target_leaf, current_path):
            if current_node == target_leaf:
                return current_path, True
            
            if tree.tree_.children_left[current_node] == tree.tree_.children_right[current_node]:
                return current_path, False  # Reached a different leaf
            
            feature_idx = tree.tree_.feature[current_node]
            threshold = tree.tree_.threshold[current_node]
            
            if feature_idx < 0:
                return current_path, False
            
            feature_name = self.feature_names[feature_idx] if feature_idx < len(self.feature_names) else f"feature_{feature_idx}"
            
            # Try left child
            left_path, found = find_path_to_leaf(
                tree.tree_.children_left[current_node], 
                target_leaf, 
                current_path + [(feature_name, threshold, "<=")]
            )
            if found:
                return left_path, True
            
            # Try right child
            right_path, found = find_path_to_leaf(
                tree.tree_.children_right[current_node],
                target_leaf,
                current_path + [(feature_name, threshold, ">")]
            )
            if found:
                return right_path, True
            
            return current_path, False
        
        path, found = find_path_to_leaf(node_id, target_leaf, [])
        if found and len(path) > 0:
            conditions = []
            for feature_name, threshold, op in path:
                if op == "<=":
                    conditions.append(f"{feature_name} <= {threshold:.4f}")
                else:
                    conditions.append(f"{feature_name} > {threshold:.4f}")
            return "IF " + " AND ".join(conditions) + f" THEN class = {target_label}"
        
        return None
    
    def _generate_statistical_rule(self, X: np.ndarray, y: np.ndarray, 
                                   target_label: int) -> str:
        """
        Generate a rule based on cluster statistics as a last resort.
        Uses feature means and standard deviations to create a simple rule.
        """
        # Find samples with target label
        label_mask = y == target_label
        if label_mask.sum() == 0:
            # If no samples with target label, use all samples
            target_X = X
        else:
            target_X = X[label_mask]
        
        # Get feature means and create simple thresholds
        feature_means = np.mean(target_X, axis=0)
        feature_stds = np.std(target_X, axis=0)
        
        # Use top 3 most variable features
        feature_variances = np.var(target_X, axis=0)
        top_features = np.argsort(feature_variances)[-3:][::-1]
        
        conditions = []
        for feat_idx in top_features:
            if feat_idx < len(self.feature_names):
                feature_name = self.feature_names[feat_idx]
                mean_val = feature_means[feat_idx]
                # Create a simple threshold rule
                conditions.append(f"{feature_name} <= {mean_val:.4f}")
        
        if conditions:
            return "IF " + " AND ".join(conditions) + f" THEN class = {target_label}"
        else:
            # Absolute fallback
            return f"IF (cluster conditions) THEN class = {target_label}"
    
    def _paths_to_rule(self, paths: List[List[Tuple[str, float, str]]], 
                      target_label: int, minimal: bool = False) -> str:
        """Convert tree paths to human-readable rule string.
        
        Never returns vague rules - raises error if paths are empty.
        """
        if not paths or len(paths) == 0:
            raise ValueError(f"Cannot generate rule: No valid paths found for target_label {target_label}")
        
        # Use the first (simplest) path for now
        # In production, you might want to find common conditions across paths
        path = paths[0]
        
        if not path or len(path) == 0:
            raise ValueError(f"Cannot generate rule: Empty path for target_label {target_label}")
        
        conditions = []
        for feature_name, threshold, op in path:
            if op == "<=":
                conditions.append(f"{feature_name} <= {threshold:.4f}")
            else:
                conditions.append(f"{feature_name} > {threshold:.4f}")
        
        rule = "IF " + " AND ".join(conditions) + f" THEN class = {target_label}"
        return rule
    
    def _find_counterfactual_rule(self, cluster_data: Dict, current_cluster_id: int,
                                  current_features: np.ndarray, current_label: int,
                                  X_train: np.ndarray, y_train: np.ndarray) -> Optional[str]:
        """Find counterfactual rule from nearest opposing cluster."""
        # Find clusters with different dominant labels
        opposing_clusters = []
        for cluster_id in np.unique(cluster_data['cluster_labels']):
            if cluster_id == current_cluster_id:
                continue
            
            cluster_mask = cluster_data['cluster_labels'] == cluster_id
            if cluster_mask.sum() < self.min_samples_split:
                continue
            
            cluster_labels = cluster_data['predictions'][cluster_mask]
            dominant_label = np.bincount(cluster_labels).argmax()
            
            if dominant_label != current_label:
                opposing_clusters.append({
                    'cluster_id': cluster_id,
                    'features': cluster_data['features'][cluster_mask],
                    'label': dominant_label,
                    'n_samples': cluster_mask.sum()
                })
        
        if not opposing_clusters:
            return None
        
        # Find nearest opposing cluster (by feature distance)
        current_mean = current_features.mean(axis=0)
        min_dist = np.inf
        nearest_cluster = None
        
        for opp_cluster in opposing_clusters:
            opp_mean = opp_cluster['features'].mean(axis=0)
            dist = np.linalg.norm(current_mean - opp_mean)
            if dist < min_dist:
                min_dist = dist
                nearest_cluster = opp_cluster
        
        if nearest_cluster is None:
            return None
        
        # Extract rule from nearest opposing cluster
        tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=42
        )
        tree.fit(nearest_cluster['features'], 
                np.full(len(nearest_cluster['features']), nearest_cluster['label']))
        
        counterfactual_rule = self._extract_rule_from_tree(
            tree, nearest_cluster['features'], nearest_cluster['label'], 
            is_explanation=False
        )
        
        return counterfactual_rule
    
    def extract_local_rules(self, influence_results: Dict, test_id: int,
                           X_train: np.ndarray, y_train: np.ndarray,
                           X_test: Optional[np.ndarray] = None,
                           top_k: int = 50, clustering_method: str = 'kmeans') -> Dict:
        """
        Complete pipeline: Extract local rules for a test instance.
        
        Args:
            influence_results: Results from calc_influence_dataset
            test_id: Test sample ID (index in test set)
            X_train: Training feature matrix
            y_train: Training labels
            X_test: Test feature matrix (optional, if None, assumes test_id is in training set)
            top_k: Number of top influential samples
            clustering_method: Clustering algorithm to use
            
        Returns:
            Dictionary with all extracted rules and metadata
        """
        # Step 1: Get influential samples
        helpful_indices, harmful_indices, helpful_infl, harmful_infl = \
            self.get_influential_samples(influence_results, test_id, top_k)
        
        # Step 2: Cluster influential samples
        cluster_data = self.cluster_influential_samples(
            X_train, helpful_indices, harmful_indices,
            helpful_infl, harmful_infl, clustering_method
        )
        
        # Step 3: Extract rules from each cluster
        rules = {}
        for cluster_id in np.unique(cluster_data['cluster_labels']):
            if cluster_id == -1:  # Skip noise cluster from DBSCAN
                continue
            rules[cluster_id] = self.extract_rules_from_cluster(
                cluster_data, cluster_id, X_train, y_train
            )
        
        # Get test sample prediction
        if X_test is not None and test_id < len(X_test):
            test_features = X_test[test_id:test_id+1]
        else:
            # Fallback: assume test_id refers to training set (for backward compatibility)
            test_features = X_train[test_id:test_id+1] if test_id < len(X_train) else None
        
        if test_features is None:
            # Use the result from influence_results
            test_pred = np.array([influence_results[str(test_id)]['label']])
        else:
            test_pred, _ = self.get_model_predictions(test_features)
        
        return {
            'test_id': test_id,
            'test_prediction': int(test_pred[0]),
            'helpful_samples': helpful_indices.tolist(),
            'harmful_samples': harmful_indices.tolist(),
            'n_clusters': len(rules),
            'clusters': rules,
            'cluster_data': {
                'cluster_labels': cluster_data['cluster_labels'].tolist(),
                'influence_signs': cluster_data['influence_signs'].tolist()
            }
        }
    
    def extract_local_rules_from_csv(self, csv_path: str, test_id: int,
                                    X_train: np.ndarray, y_train: np.ndarray,
                                    X_test: Optional[np.ndarray] = None,
                                    top_k: int = 50, clustering_method: str = 'kmeans') -> Dict:
        """
        Complete pipeline: Extract local rules for a test instance from CSV file.
        
        Args:
            csv_path: Path to CSV file with columns: train_id, test_id, influence
            test_id: Test sample ID (index in test set)
            X_train: Training feature matrix
            y_train: Training labels
            X_test: Test feature matrix (optional, if None, assumes test_id is in training set)
            top_k: Number of top influential samples
            clustering_method: Clustering algorithm to use
            
        Returns:
            Dictionary with all extracted rules and metadata
        """
        # Step 1: Get influential samples from CSV
        helpful_indices, harmful_indices, helpful_infl, harmful_infl = \
            self.get_influential_samples_from_csv(csv_path, test_id, top_k)
        
        # Step 2: Cluster influential samples
        cluster_data = self.cluster_influential_samples(
            X_train, helpful_indices, harmful_indices,
            helpful_infl, harmful_infl, clustering_method
        )
        
        # Step 3: Extract rules from each cluster
        rules = {}
        for cluster_id in np.unique(cluster_data['cluster_labels']):
            if cluster_id == -1:  # Skip noise cluster from DBSCAN
                continue
            rules[cluster_id] = self.extract_rules_from_cluster(
                cluster_data, cluster_id, X_train, y_train
            )
        
        # Get test sample prediction
        if X_test is not None and test_id < len(X_test):
            test_features = X_test[test_id:test_id+1]
        else:
            # Fallback: assume test_id refers to training set (for backward compatibility)
            test_features = X_train[test_id:test_id+1] if test_id < len(X_train) else None
        
        if test_features is None:
            # Try to get prediction from model using a sample
            # This shouldn't happen in normal usage
            test_pred = np.array([0])  # Default
        else:
            test_pred, _ = self.get_model_predictions(test_features)
        
        return {
            'test_id': test_id,
            'test_prediction': int(test_pred[0]),
            'helpful_samples': helpful_indices.tolist(),
            'harmful_samples': harmful_indices.tolist(),
            'n_clusters': len(rules),
            'clusters': rules,
            'cluster_data': {
                'cluster_labels': cluster_data['cluster_labels'].tolist(),
                'influence_signs': cluster_data['influence_signs'].tolist()
            }
        }


    def visualize_decision_tree(self, tree: DecisionTreeClassifier, cluster_id: int,
                                feature_names: Optional[List[str]] = None,
                                max_depth: int = None, figsize=(20, 10)):
        """Visualize the decision tree for a cluster.
        
        Args:
            tree: Fitted DecisionTreeClassifier
            cluster_id: ID of the cluster
            feature_names: List of feature names (uses self.feature_names if None)
            max_depth: Maximum depth to show (uses tree.max_depth if None)
            figsize: Figure size tuple
        """
        if feature_names is None:
            feature_names = self.feature_names
        
        if max_depth is None:
            max_depth = tree.max_depth if hasattr(tree, 'max_depth') else 5
        
        fig, ax = plt.subplots(figsize=figsize)
        plot_tree(tree, feature_names=feature_names, 
                 class_names=[f"Class {i}" for i in range(tree.n_classes_)],
                 filled=True, rounded=True, fontsize=10, max_depth=max_depth, ax=ax)
        ax.set_title(f"Decision Tree for Cluster {cluster_id}", fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def visualize_sample_path(self, tree: DecisionTreeClassifier, sample: np.ndarray,
                              sample_label: int, cluster_id: int,
                              feature_names: Optional[List[str]] = None):
        """Visualize the path a sample takes through the decision tree.
        
        Args:
            tree: Fitted DecisionTreeClassifier
            sample: Single sample (1D array)
            sample_label: True label of the sample
            cluster_id: ID of the cluster
            feature_names: List of feature names (uses self.feature_names if None)
        
        Returns:
            Dictionary with path information and visualization
        """
        if feature_names is None:
            feature_names = self.feature_names
        
        sample = sample.reshape(1, -1)
        prediction = tree.predict(sample)[0]
        decision_path = tree.decision_path(sample)
        node_indicator = decision_path.toarray()[0]
        
        # Get path nodes
        path_nodes = np.where(node_indicator == 1)[0]
        leaf_node = tree.apply(sample)[0]
        
        # Extract path conditions
        path_conditions = []
        for node_id in path_nodes:
            if node_id == leaf_node:
                break
            if tree.tree_.children_left[node_id] != tree.tree_.children_right[node_id]:  # Not a leaf
                feature_idx = tree.tree_.feature[node_id]
                threshold = tree.tree_.threshold[node_id]
                if feature_idx >= 0:
                    feature_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f"feature_{feature_idx}"
                    sample_value = sample[0][feature_idx]
                    if sample_value <= threshold:
                        path_conditions.append({
                            'node': node_id,
                            'feature': feature_name,
                            'threshold': threshold,
                            'sample_value': sample_value,
                            'condition': f"{feature_name} <= {threshold:.4f}",
                            'direction': 'left'
                        })
                    else:
                        path_conditions.append({
                            'node': node_id,
                            'feature': feature_name,
                            'threshold': threshold,
                            'sample_value': sample_value,
                            'condition': f"{feature_name} > {threshold:.4f}",
                            'direction': 'right'
                        })
        
        # Get leaf node info
        leaf_value = tree.tree_.value[leaf_node][0]
        leaf_class = np.argmax(leaf_value)
        leaf_samples = int(leaf_value[leaf_class])
        
        return {
            'cluster_id': cluster_id,
            'sample': sample[0].tolist(),
            'true_label': int(sample_label),
            'predicted_label': int(prediction),
            'leaf_node': int(leaf_node),
            'path_nodes': path_nodes.tolist(),
            'path_conditions': path_conditions,
            'leaf_class': int(leaf_class),
            'leaf_samples': int(leaf_samples),
            'path_string': " → ".join([c['condition'] for c in path_conditions]) + f" → Class {leaf_class}"
        }
    
    def print_tree_text(self, tree: DecisionTreeClassifier, 
                       feature_names: Optional[List[str]] = None,
                       max_depth: int = None):
        """Print the decision tree in text format.
        
        Args:
            tree: Fitted DecisionTreeClassifier
            feature_names: List of feature names (uses self.feature_names if None)
            max_depth: Maximum depth to show
        """
        if feature_names is None:
            feature_names = self.feature_names
        
        if max_depth is None:
            max_depth = tree.max_depth if hasattr(tree, 'max_depth') else 10
        
        tree_text = export_text(tree, feature_names=feature_names, max_depth=max_depth)
        print(tree_text)
        return tree_text
