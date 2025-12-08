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
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import torch


class LocalRuleExtractor:
    """
    Extracts local symbolic rules from TabNet predictions using influence functions.
    
    Pipeline:
    1. Identify influential training samples (S_plus, S_minus)
    2. Cluster influential samples by features, logits, and influence sign
    3. Derive local symbolic rules from each cluster using decision trees
    """
    
    def __init__(self, model, feature_names: List[str], n_clusters: int = 3, 
                 max_depth: int = 5, min_samples_split: int = 10):
        """
        Args:
            model: Trained TabNet model (wrapped with TabNetWrapper)
            feature_names: List of feature names for rule interpretation
            n_clusters: Number of clusters for influential samples
            max_depth: Maximum depth for decision trees
            min_samples_split: Minimum samples to split a node
        """
        self.model = model
        self.feature_names = feature_names
        self.n_clusters = n_clusters
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.model.eval()
    
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
        influences = np.array(result['influence'])
        helpful_indices = np.array(result['helpful'][:top_k])
        harmful_indices = np.array(result['harmful'][:top_k])
        
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
        
        # Create clustering features: combine normalized features, logits, and influence
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(all_features)
        logits_scaled = StandardScaler().fit_transform(all_logits)
        influences_scaled = StandardScaler().fit_transform(all_influences.reshape(-1, 1))
        
        # Combine features for clustering
        clustering_features = np.hstack([
            features_scaled,
            logits_scaled,
            influences_scaled,
            influence_signs.reshape(-1, 1)
        ])
        
        # Perform clustering
        if clustering_method == 'kmeans':
            clusterer = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            cluster_labels = clusterer.fit_predict(clustering_features)
        elif clustering_method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = clusterer.fit_predict(clustering_features)
        else:
            raise ValueError(f"Unknown clustering method: {clustering_method}")
        
        return {
            'cluster_labels': cluster_labels,
            'features': all_features,
            'logits': all_logits,
            'predictions': all_labels,
            'influences': all_influences,
            'indices': all_indices,
            'influence_signs': influence_signs,
            'scaler': scaler,
            'clusterer': clusterer
        }
    
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
        explanation_rule = self._extract_rule_from_tree(tree, cluster_features, 
                                                      most_common_label, 
                                                      is_explanation=True)
        
        # Find counterfactual: nearest cluster with different dominant label
        counterfactual_rule = self._find_counterfactual_rule(
            cluster_data, cluster_id, cluster_features, most_common_label,
            X_train, y_train
        )
        
        return {
            'cluster_id': cluster_id,
            'n_samples': cluster_mask.sum(),
            'dominant_label': most_common_label,
            'mean_influence': cluster_influences.mean(),
            'explanation_rule': explanation_rule,
            'counterfactual_rule': counterfactual_rule,
            'tree': tree,
            'cluster_indices': cluster_indices.tolist()
        }
    
    def _extract_rule_from_tree(self, tree: DecisionTreeClassifier, 
                                X: np.ndarray, target_label: int,
                                is_explanation: bool = True) -> str:
        """Extract a human-readable rule from decision tree."""
        # Get tree structure
        tree_text = export_text(tree, feature_names=self.feature_names, 
                               max_depth=self.max_depth)
        
        # Find paths leading to target label
        paths = []
        for i in range(len(X)):
            pred = tree.predict([X[i]])[0]
            if pred == target_label:
                # Trace path through tree
                path = self._trace_tree_path(tree, X[i], target_label)
                if path:
                    paths.append(path)
        
        if not paths:
            return f"IF (complex conditions) THEN class = {target_label}"
        
        # Use most common path or simplest path
        if is_explanation:
            # For explanation, use the most common path
            path_str = self._paths_to_rule(paths, target_label)
        else:
            # For counterfactual, use minimal change path
            path_str = self._paths_to_rule(paths, target_label, minimal=True)
        
        return path_str
    
    def _trace_tree_path(self, tree: DecisionTreeClassifier, sample: np.ndarray,
                        target_label: int) -> Optional[List[Tuple[str, float, str]]]:
        """Trace a path through the decision tree for a given sample."""
        node = 0
        path = []
        
        while True:
            if tree.tree_.children_left[node] == tree.tree_.children_right[node]:
                # Leaf node
                if tree.tree_.value[node][0].argmax() == target_label:
                    return path
                return None
            
            feature_idx = tree.tree_.feature[node]
            threshold = tree.tree_.threshold[node]
            feature_name = self.feature_names[feature_idx] if feature_idx < len(self.feature_names) else f"feature_{feature_idx}"
            
            if sample[feature_idx] <= threshold:
                path.append((feature_name, threshold, "<="))
                node = tree.tree_.children_left[node]
            else:
                path.append((feature_name, threshold, ">"))
                node = tree.tree_.children_right[node]
    
    def _paths_to_rule(self, paths: List[List[Tuple[str, float, str]]], 
                      target_label: int, minimal: bool = False) -> str:
        """Convert tree paths to human-readable rule string."""
        if not paths:
            return f"IF (complex) THEN class = {target_label}"
        
        # Use the first (simplest) path for now
        # In production, you might want to find common conditions across paths
        path = paths[0]
        
        conditions = []
        for feature_name, threshold, op in path:
            if op == "<=":
                conditions.append(f"{feature_name} <= {threshold:.2f}")
            else:
                conditions.append(f"{feature_name} > {threshold:.2f}")
        
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

