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
    node_colors = {}
    edge_labels = {}
    
    def calculate_subtree_size(node, depth=0):
        """Calculate the width (number of leaves) of subtree."""
        if depth > max_depth:
            return 0
        if node is None:
            return 0
        if node.is_leaf():
            return 1
        left_size = calculate_subtree_size(node.left, depth + 1) if node.left else 0
        right_size = calculate_subtree_size(node.right, depth + 1) if node.right else 0
        return max(left_size, 1) + max(right_size, 1)
    
    def build_tree_and_positions(node, parent_id=None, depth=0, x_offset=0, level_width=1.0):
        """
        Build graph and calculate positions for binary tree layout.
        Returns: (x_position, subtree_width)
        """
        if depth > max_depth or node is None:
            return x_offset, 0
        
        node_id = node.node_id
        G.add_node(node_id)
        
        # Calculate subtree sizes for spacing
        left_size = calculate_subtree_size(node.left, depth + 1) if node.left else 0
        right_size = calculate_subtree_size(node.right, depth + 1) if node.right else 0
        total_size = max(left_size + right_size, 1)
        
        # Calculate x position (centered between children)
        if node.is_leaf():
            # Leaf node: position at current offset
            x_pos = x_offset
            y_pos = -depth * 2.5  # Vertical spacing between levels
            
            # Create label
            label = f"Class {node.label}"
            node_labels[node_id] = label
            node_colors[node_id] = 'lightgreen' if node.label == 1 else 'lightcoral'
            pos[node_id] = (x_pos, y_pos)
            
            if parent_id is not None:
                G.add_edge(parent_id, node_id)
            
            return x_pos, 1
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
            node_colors[node_id] = 'lightblue'
            
            # Traverse left subtree
            left_x, left_width = build_tree_and_positions(
                node.left, node_id, depth + 1, x_offset, level_width
            )
            
            # Calculate x position (between left and right subtrees)
            right_x_start = x_offset + max(left_width, 0.5)
            right_x, right_width = build_tree_and_positions(
                node.right, node_id, depth + 1, right_x_start, level_width
            )
            
            # Position current node between children
            if node.left and node.right:
                x_pos = (left_x + right_x) / 2
            elif node.left:
                x_pos = left_x
            elif node.right:
                x_pos = right_x
            else:
                x_pos = x_offset
            
            y_pos = -depth * 2.5
            pos[node_id] = (x_pos, y_pos)
            
            # Add edges with labels
            if parent_id is not None:
                G.add_edge(parent_id, node_id)
            
            if node.left:
                G.add_edge(node_id, node.left.node_id)
                edge_labels[(node_id, node.left.node_id)] = "Yes"  # Rule matches
            if node.right:
                G.add_edge(node_id, node.right.node_id)
                edge_labels[(node_id, node.right.node_id)] = "No"   # Rule doesn't match
            
            return x_pos, left_width + right_width
    
    # Build graph and calculate positions
    build_tree_and_positions(tree_root)
    
    # Normalize positions to fit in figure
    if pos:
        x_coords = [p[0] for p in pos.values()]
        y_coords = [p[1] for p in pos.values()]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add padding
        x_range = x_max - x_min if x_max != x_min else 1
        y_range = y_max - y_min if y_max != y_min else 1
        
        # Normalize to [0, 1] range with padding
        normalized_pos = {}
        for node_id, (x, y) in pos.items():
            normalized_x = (x - x_min) / x_range * 0.9 + 0.05  # 5% padding on each side
            normalized_y = (y - y_min) / y_range * 0.9 + 0.05
            normalized_pos[node_id] = (normalized_x, normalized_y)
        
        pos = normalized_pos
    
    # Create visualization with proper binary tree layout
    fig, ax = plt.subplots(figsize=(24, 16))
    
    # Calculate figure bounds based on positions
    if pos:
        x_coords = [p[0] for p in pos.values()]
        y_coords = [p[1] for p in pos.values()]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add margins
        x_margin = (x_max - x_min) * 0.1 if x_max != x_min else 1
        y_margin = (y_max - y_min) * 0.1 if y_max != y_min else 1
        
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
    else:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    
    # Draw edges first (so nodes appear on top)
    for edge in G.edges():
        if edge[0] in pos and edge[1] in pos:
            x1, y1 = pos[edge[0]]
            x2, y2 = pos[edge[1]]
            
            # Draw edge as arrow
            arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                   arrowstyle='->', mutation_scale=25,
                                   color='gray', linewidth=2.5, alpha=0.7, zorder=1)
            ax.add_patch(arrow)
            
            # Add edge label (Yes/No)
            if edge in edge_labels:
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                # Offset label perpendicular to edge (adjusted for smaller nodes)
                dx, dy = x2 - x1, y2 - y1
                length = np.sqrt(dx**2 + dy**2)
                if length > 0:
                    # Perpendicular direction (smaller offset for smaller nodes)
                    perp_x = -dy / length * 0.02
                    perp_y = dx / length * 0.02
                    label_x = mid_x + perp_x
                    label_y = mid_y + perp_y
                else:
                    label_x, label_y = mid_x, mid_y
                
                ax.text(label_x, label_y, edge_labels[edge], 
                       fontsize=8, ha='center', va='center', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', 
                                edgecolor='black', alpha=0.9, linewidth=1.2),
                       zorder=5)
    
    # Draw nodes as rectangular boxes (like traditional decision trees)
    # Smaller node size with text that fits
    node_width = 0.05
    node_height = 0.04
    
    for node_id in G.nodes():
        if node_id in pos:
            x, y = pos[node_id]
            color = node_colors.get(node_id, 'lightgray')
            label = node_labels.get(node_id, str(node_id))
            
            # Draw node as rounded rectangle
            node_box = FancyBboxPatch((x - node_width/2, y - node_height/2), 
                                     node_width, node_height,
                                     boxstyle="round,pad=0.005", 
                                     facecolor=color, 
                                     edgecolor='black', linewidth=2, 
                                     zorder=3, alpha=0.9)
            ax.add_patch(node_box)
            
            # Draw label inside node with smaller font
            # Adjust font size based on label length to ensure it fits
            label_lines = label.split('\n')
            num_lines = len(label_lines)
            max_line_length = max(len(line) for line in label_lines) if label_lines else len(label)
            
            # Calculate appropriate font size to fit in smaller node
            # Smaller nodes need smaller fonts
            if num_lines <= 2 and max_line_length <= 12:
                font_size = 7
            elif num_lines <= 3 and max_line_length <= 18:
                font_size = 6
            elif num_lines <= 4:
                font_size = 5
            else:
                font_size = 4
            
            # Truncate very long labels if necessary
            if max_line_length > 25:
                truncated_lines = []
                for line in label_lines:
                    if len(line) > 25:
                        truncated_lines.append(line[:22] + '...')
                    else:
                        truncated_lines.append(line)
                label = '\n'.join(truncated_lines)
            
            ax.text(x, y, label, ha='center', va='center',
                   fontsize=font_size, fontweight='bold', zorder=4,
                   color='black' if color not in ['lightgreen', 'lightcoral'] else 'white')
    
    ax.set_title("Global Rules Decision Tree\n(Binary Tree Structure)", 
                fontsize=20, fontweight='bold', pad=30)
    ax.axis('off')
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    return fig, G


def _hierarchical_tree_layout(G: 'nx.DiGraph', root: RuleTreeNode, max_depth: int) -> dict:
    """Create a hierarchical tree layout."""
    pos = {}
    
    def assign_positions(node, depth=0, x=0, y=0, width=10):
        if depth > max_depth:
            return x
        
        node_id = node.node_id
        if node_id not in G.nodes():
            return x
        
        # Calculate y position (depth)
        y_pos = -depth * 2
        
        # Calculate x position based on subtree width
        if node.is_leaf():
            pos[node_id] = (x, y_pos)
            return x + 1
        else:
            # Get children
            left_x = x
            if node.left:
                left_x = assign_positions(node.left, depth + 1, x, y_pos, width * 0.5)
            
            # Position current node between children
            if node.right:
                right_x = assign_positions(node.right, depth + 1, left_x + 0.5, y_pos, width * 0.5)
                current_x = (left_x + right_x) / 2
            else:
                current_x = left_x
            
            pos[node_id] = (current_x, y_pos)
            return max(left_x, right_x if node.right else left_x) + 1
    
    assign_positions(root)
    return pos


def _prevent_overlapping(pos: dict, node_labels: dict, min_distance: float = 0.5) -> dict:
    """Adjust positions to prevent overlapping nodes."""
    import numpy as np
    
    pos_array = np.array([pos[node] for node in pos.keys()])
    nodes = list(pos.keys())
    
    # Check for overlaps and adjust
    max_iterations = 50
    for iteration in range(max_iterations):
        overlaps_found = False
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes[i+1:], i+1):
                dist = np.linalg.norm(pos_array[i] - pos_array[j])
                if dist < min_distance:
                    overlaps_found = True
                    # Move nodes apart
                    direction = pos_array[j] - pos_array[i]
                    if np.linalg.norm(direction) < 1e-6:
                        direction = np.array([1, 0])  # Default direction
                    direction = direction / np.linalg.norm(direction)
                    
                    # Move both nodes
                    move_distance = (min_distance - dist) / 2
                    pos_array[i] -= direction * move_distance
                    pos_array[j] += direction * move_distance
        
        if not overlaps_found:
            break
    
    # Update positions
    new_pos = {node: tuple(pos_array[i]) for i, node in enumerate(nodes)}
    return new_pos


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

