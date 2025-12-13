#! /usr/bin/env python3
"""
Rule Tree: Exact conversion from ordered rule list to binary decision tree.

This module converts an ordered list of global rules into a binary decision tree
that preserves the exact semantics of the rule list (nested-if structure),
ensuring surrogate fidelity = 1.
"""

import numpy as np
from typing import List, Optional
from utilities.influence.global_rule_aggregation import LocalRule, GlobalRuleAggregator


class RuleTreeNode:
    """Node in a decision tree built from ordered global rules."""
    def __init__(self, rule=None, label=None, left=None, right=None, node_id=None):
        self.rule = rule  # LocalRule object (None for leaf nodes)
        self.label = label  # Class label (for leaf nodes)
        self.left = left  # Left child (rule matches)
        self.right = right  # Right child (rule doesn't match)
        self.node_id = node_id  # Unique identifier
    
    def is_leaf(self):
        return self.rule is None and self.label is not None


def build_rule_tree(global_rules: List[LocalRule], aggregator: GlobalRuleAggregator, 
                   feature_names: List[str], default_label: int = 0, 
                   expand_conjunctions: bool = True) -> RuleTreeNode:
    """
    Build a binary decision tree from an ordered list of global rules.
    
    Algorithm:
    - Root checks first rule's condition (conjunction)
    - Left child = leaf with rule's label
    - Right child = subtree for remaining rules (recursive)
    - Optionally expand conjunctions to binary splits on single features
    
    Args:
        global_rules: Ordered list of LocalRule objects
        aggregator: GlobalRuleAggregator instance
        feature_names: List of feature names
        default_label: Default prediction if no rule matches
        expand_conjunctions: If True, expand conjunctions to binary splits on single features
    
    Returns:
        root: RuleTreeNode root of the tree
    """
    if len(global_rules) == 0:
        return RuleTreeNode(label=default_label, node_id=0)
    
    class NodeIdCounter:
        def __init__(self):
            self.count = 0
        def __iter__(self):
            return self
        def __next__(self):
            self.count += 1
            return self.count
    
    def build_recursive(rules, node_id_counter, default):
        """Recursively build tree from remaining rules."""
        if len(rules) == 0:
            return RuleTreeNode(label=default, node_id=next(node_id_counter)), node_id_counter
        
        current_rule = rules[0]
        remaining_rules = rules[1:]
        
        if expand_conjunctions and len(current_rule.premises) > 1:
            # Expand conjunction to a chain of binary splits
            premises_list = list(current_rule.premises.items())
            
            def build_conjunction_chain(premises, remaining_rules, node_id_counter, default):
                """Build a chain of nodes for conjunction."""
                if len(premises) == 0:
                    # All conditions met, return leaf with rule's label
                    return RuleTreeNode(label=current_rule.consequence, node_id=next(node_id_counter)), node_id_counter
                
                feature_name, (lower, upper) = premises[0]
                remaining_premises = premises[1:]
                
                node_id = next(node_id_counter)
                
                # Left child: condition met, check remaining premises
                left_child, node_id_counter = build_conjunction_chain(
                    remaining_premises, remaining_rules, node_id_counter, default
                )
                
                # Right child: condition not met, check remaining rules
                right_child, node_id_counter = build_recursive(remaining_rules, node_id_counter, default)
                
                # Create intermediate node (stores feature check info)
                node = RuleTreeNode(
                    rule=LocalRule(
                        premises={feature_name: (lower, upper)},
                        consequence=current_rule.consequence,
                        rule_string=f"{feature_name} in [{lower:.4f}, {upper:.4f}]"
                    ),
                    left=left_child,
                    right=right_child,
                    node_id=node_id
                )
                
                return node, node_id_counter
            
            root, node_id_counter = build_conjunction_chain(
                premises_list, remaining_rules, node_id_counter, default
            )
            return root, node_id_counter
        else:
            # Single condition or don't expand: check entire rule at once
            node_id = next(node_id_counter)
            
            # Left child: rule matches -> leaf with rule's label
            left_child = RuleTreeNode(label=current_rule.consequence, node_id=next(node_id_counter))
            
            # Right child: rule doesn't match -> subtree for remaining rules
            right_child, node_id_counter = build_recursive(remaining_rules, node_id_counter, default)
            
            root = RuleTreeNode(
                rule=current_rule,
                left=left_child,
                right=right_child,
                node_id=node_id
            )
            
            return root, node_id_counter
    
    node_id_counter = NodeIdCounter()
    root, _ = build_recursive(global_rules, node_id_counter, default_label)
    return root


def predict_with_rule_tree(tree_root: RuleTreeNode, aggregator: GlobalRuleAggregator, 
                          X: np.ndarray, feature_names: List[str]) -> tuple:
    """
    Make predictions using a rule tree.
    
    Args:
        tree_root: RuleTreeNode root of the tree
        aggregator: GlobalRuleAggregator instance
        X: Input data (n_samples, n_features)
        feature_names: List of feature names
    
    Returns:
        predictions: Array of predictions (n_samples,)
        rule_coverage: Array indicating which rule matched each sample (n_samples,)
    """
    n_samples = X.shape[0]
    predictions = np.full(n_samples, -1, dtype=int)
    rule_coverage = np.full(n_samples, -1, dtype=int)
    
    def predict_sample(node, sample, sample_idx):
        """Recursively traverse tree to predict a single sample."""
        if node.is_leaf():
            predictions[sample_idx] = node.label
            rule_coverage[sample_idx] = node.node_id
            return
        
        # Check if rule condition is met
        if node.rule is not None:
            coverage = aggregator.coverage(node.rule, sample.reshape(1, -1))
            if coverage[0]:
                # Condition met: go left
                predict_sample(node.left, sample, sample_idx)
            else:
                # Condition not met: go right
                predict_sample(node.right, sample, sample_idx)
        else:
            # Should not happen, but handle gracefully
            predictions[sample_idx] = node.label if node.label is not None else 0
            rule_coverage[sample_idx] = node.node_id
    
    for sample_idx in range(n_samples):
        predict_sample(tree_root, X[sample_idx], sample_idx)
    
    return predictions, rule_coverage


def visualize_rule_tree(tree_root: RuleTreeNode, aggregator: GlobalRuleAggregator,
                        feature_names: List[str], X: Optional[np.ndarray] = None,
                        y: Optional[np.ndarray] = None, max_depth: int = 10):
    """
    Visualize the rule tree using networkx and matplotlib.
    
    Args:
        tree_root: RuleTreeNode root of the tree
        aggregator: GlobalRuleAggregator instance
        feature_names: List of feature names
        X: Training data (optional, for computing coverage/fidelity)
        y: Training labels (optional, for computing fidelity)
        max_depth: Maximum depth to visualize
    
    Returns:
        fig: matplotlib figure
        G: networkx graph
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    
    G = nx.DiGraph()
    pos = {}
    node_labels = {}
    node_colors = []
    
    def traverse_tree(node, parent_id=None, depth=0, x_pos=0, y_pos=0, x_offset=1.0):
        """Traverse tree and build graph."""
        if depth > max_depth:
            return x_pos
        
        node_id = node.node_id
        
        # Add node to graph
        G.add_node(node_id)
        
        # Calculate position
        if node.is_leaf():
            # Leaf node
            label = f"Class {node.label}"
            node_labels[node_id] = label
            node_colors.append('lightgreen' if node.label == 1 else 'lightcoral')
            pos[node_id] = (x_pos, y_pos)
            
            if parent_id is not None:
                G.add_edge(parent_id, node_id, label="✓")
            
            return x_pos
        else:
            # Internal node
            rule = node.rule
            if rule is not None:
                # Create label from rule
                if len(rule.premises) == 1:
                    feature, (lower, upper) = list(rule.premises.items())[0]
                    if lower == float('-inf'):
                        label = f"{feature}\n≤ {upper:.2f}"
                    elif upper == float('inf'):
                        label = f"{feature}\n> {lower:.2f}"
                    else:
                        label = f"{lower:.2f} < {feature}\n≤ {upper:.2f}"
                else:
                    label = f"Rule {node_id}\n{len(rule.premises)} features"
                
                # Add coverage/fidelity info if available
                if X is not None:
                    coverage = aggregator.coverage(rule, X)
                    coverage_pct = (coverage.sum() / len(X)) * 100
                    label += f"\nCoverage: {coverage_pct:.1f}%"
                    if y is not None:
                        fidelity = aggregator.binary_fidelity(rule, X, y)
                        label += f"\nFidelity: {fidelity:.3f}"
            else:
                label = f"Node {node_id}"
            
            node_labels[node_id] = label
            node_colors.append('lightblue')
            pos[node_id] = (x_pos, y_pos)
            
            if parent_id is not None:
                G.add_edge(parent_id, node_id)
            
            # Traverse children
            left_x = traverse_tree(node.left, node_id, depth + 1, 
                                   x_pos - x_offset, y_pos - 1, x_offset * 0.6)
            right_x = traverse_tree(node.right, node_id, depth + 1,
                                    left_x + x_offset * 0.3, y_pos - 1, x_offset * 0.6)
            
            return max(left_x, right_x)
    
    # Build graph
    traverse_tree(tree_root)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                           node_size=2000, alpha=0.8, ax=ax)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.6, arrows=True, 
                          arrowsize=20, edge_color='gray', ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, node_labels, font_size=8, ax=ax)
    
    ax.set_title("Global Rules Decision Tree\n(Exact conversion from ordered rule list)", 
                fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    return fig, G


def predict_with_global_rules(global_rules: List[LocalRule], aggregator: GlobalRuleAggregator,
                             X: np.ndarray, feature_names: List[str], 
                             default_label: Optional[int] = None) -> tuple:
    """
    Predict using global rules via a decision tree (exact conversion from ordered rule list).
    
    This function builds a binary decision tree from the ordered global rules that preserves
    the exact semantics of the rule list (nested-if structure), ensuring surrogate fidelity = 1.
    
    Args:
        global_rules: Ordered list of LocalRule objects
        aggregator: GlobalRuleAggregator instance
        X: Input data (n_samples, n_features)
        feature_names: List of feature names
        default_label: Default prediction if no rule matches (defaults to majority class)
    
    Returns:
        predictions: Array of predictions (n_samples,)
        rule_coverage: Array indicating which rule matched each sample (n_samples,)
    """
    # Determine default label if not provided
    if default_label is None:
        # Try to get from globals (for notebook compatibility)
        import sys
        frame = sys._getframe(1)
        if 'y_train' in frame.f_globals:
            y_train = frame.f_globals['y_train']
            default_label = int(np.bincount(y_train).argmax())
        else:
            default_label = 0
    
    # Build tree from ordered rules
    tree_root = build_rule_tree(global_rules, aggregator, feature_names, default_label, expand_conjunctions=True)
    
    # Make predictions using tree
    predictions, rule_coverage = predict_with_rule_tree(tree_root, aggregator, X, feature_names)
    
    return predictions, rule_coverage

