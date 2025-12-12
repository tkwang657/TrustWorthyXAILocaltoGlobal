#! /usr/bin/env python3
"""
Global Rule Aggregation from Local Rules
Based on GLocalX algorithm for merging local explanations into global ones.

This module aggregates local rules extracted from influence functions into
global rules that explain the model's behavior across the entire dataset.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from functools import reduce
import re


class LocalRule:
    """Represents a local rule with premises and consequence."""
    
    def __init__(self, premises: Dict[str, Tuple[float, float]], 
                 consequence: int, rule_string: str = None):
        """
        Args:
            premises: Dictionary mapping feature names to (lower_bound, upper_bound) tuples
            consequence: Predicted class (0 or 1)
            rule_string: Original rule string for reference
        """
        self.premises = premises
        self.consequence = consequence
        self.rule_string = rule_string
        self.features = set(premises.keys())
    
    def __hash__(self):
        return hash((tuple(sorted(self.premises.items())), self.consequence))
    
    def __eq__(self, other):
        if not isinstance(other, LocalRule):
            return False
        return (self.premises == other.premises and 
                self.consequence == other.consequence)
    
    def __repr__(self):
        return f"LocalRule(consequence={self.consequence}, features={len(self.features)})"
    
    def __str__(self):
        if self.rule_string:
            return self.rule_string
        conditions = []
        for feature, (lower, upper) in self.premises.items():
            if lower == float('-inf'):
                conditions.append(f"{feature} <= {upper:.4f}")
            elif upper == float('inf'):
                conditions.append(f"{feature} > {lower:.4f}")
            else:
                conditions.append(f"{lower:.4f} < {feature} <= {upper:.4f}")
        return "IF " + " AND ".join(conditions) + f" THEN class = {self.consequence}"
    
    def covers(self, x: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Check which samples in x are covered by this rule.
        
        Matches GLocalX coverage logic: (value > lower) & (value <= upper)
        """
        if x.shape[1] != len(feature_names):
            raise ValueError(f"Feature count mismatch: x has {x.shape[1]} features, "
                           f"feature_names has {len(feature_names)}")
        
        coverage = np.ones(x.shape[0], dtype=bool)
        for feature, (lower, upper) in self.premises.items():
            if feature not in feature_names:
                continue
            feature_idx = feature_names.index(feature)
            feature_values = x[:, feature_idx]
            # GLocalX logic: (value > lower) & (value <= upper)
            # Handle infinity bounds
            if lower == float('-inf'):
                feature_coverage = feature_values <= upper
            elif upper == float('inf'):
                feature_coverage = feature_values > lower
            else:
                feature_coverage = (feature_values > lower) & (feature_values <= upper)
            coverage = coverage & feature_coverage
        
        return coverage
    
    def __and__(self, other):
        """Intersection of two rules (shared features only)."""
        if not isinstance(other, LocalRule):
            return None
        
        shared_features = self.features & other.features
        if len(shared_features) == 0:
            return None
        
        new_premises = {}
        for feature in shared_features:
            lower1, upper1 = self.premises[feature]
            lower2, upper2 = other.premises[feature]
            # Intersection: take max of lower bounds, min of upper bounds
            new_lower = max(lower1, lower2)
            new_upper = min(upper1, upper2)
            if new_lower < new_upper:  # Valid intersection
                new_premises[feature] = (new_lower, new_upper)
        
        if len(new_premises) == 0:
            return None
        
        # Use the consequence of the first rule (or could use majority vote)
        return LocalRule(new_premises, self.consequence)
    
    def __sub__(self, other):
        """Subtract other rule's features from this rule."""
        if not isinstance(other, LocalRule):
            return {self}
        
        remaining_features = self.features - other.features
        if len(remaining_features) == 0:
            return set()
        
        new_premises = {f: self.premises[f] for f in remaining_features}
        return {LocalRule(new_premises, self.consequence, self.rule_string)}


def parse_rule_string(rule_string: str, feature_names: List[str]) -> Optional[LocalRule]:
    """
    Parse a rule string like "IF feature1 <= 0.5 AND feature2 > 1.0 THEN class = 1"
    into a LocalRule object.
    """
    if not rule_string or rule_string.strip() == "":
        return None
    
    # Extract consequence
    consequence_match = re.search(r'class\s*=\s*(\d+)', rule_string)
    if not consequence_match:
        return None
    consequence = int(consequence_match.group(1))
    
    # Extract conditions
    premises = {}
    # Pattern: feature_name operator value
    # Operators: <=, <, >, >=
    condition_pattern = r'([^\s<>=]+)\s*(<=|>=|<|>)\s*([\d\.\-]+)'
    conditions = re.findall(condition_pattern, rule_string)
    
    for feature_name, operator, value_str in conditions:
        feature_name = feature_name.strip()
        value = float(value_str)
        
        if feature_name not in feature_names:
            continue
        
        if operator == '<=':
            # feature <= value means (-inf, value]
            premises[feature_name] = (float('-inf'), value)
        elif operator == '<':
            # feature < value means (-inf, value) (approximate as <= value - epsilon)
            premises[feature_name] = (float('-inf'), value - 1e-6)
        elif operator == '>':
            # feature > value means (value, inf)
            premises[feature_name] = (value, float('inf'))
        elif operator == '>=':
            # feature >= value means (value, inf) (approximate as > value - epsilon)
            premises[feature_name] = (value - 1e-6, float('inf'))
    
    if len(premises) == 0:
        return None
    
    return LocalRule(premises, consequence, rule_string)


class GlobalRuleAggregator:
    """
    Aggregates local rules into global rules using GLocalX-style merging.
    """
    
    def __init__(self, feature_names: List[str], 
                 fidelity_weight: float = 1.0,
                 complexity_weight: float = 1.0,
                 strict_join: bool = True,
                 strict_cut: bool = True):
        """
        Args:
            feature_names: List of feature names
            fidelity_weight: Weight for fidelity in BIC calculation
            complexity_weight: Weight for complexity in BIC calculation
            strict_join: If True, only merge shared features
            strict_cut: If True, cut on all features
        """
        self.feature_names = feature_names
        self.fidelity_weight = fidelity_weight
        self.complexity_weight = complexity_weight
        self.strict_join = strict_join
        self.strict_cut = strict_cut
        self.coverages = {}  # Cache for rule coverages
    
    def coverage(self, rule: LocalRule, x: np.ndarray) -> np.ndarray:
        """Get coverage of rule on data x."""
        rule_id = id(rule)
        if rule_id not in self.coverages:
            self.coverages[rule_id] = rule.covers(x, self.feature_names)
        return self.coverages[rule_id]
    
    def binary_fidelity(self, rule: LocalRule, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate binary fidelity of rule (1 - Hamming distance)."""
        coverage = self.coverage(rule, x)
        if coverage.sum() == 0:
            return 0.0
        
        rule_predictions = np.full(len(x), rule.consequence)
        rule_predictions[~coverage] = int(y.mean().round())  # Default to majority class
        
        # Hamming distance
        hamming_dist = np.mean(rule_predictions != y)
        return 1.0 - hamming_dist
    
    def bic(self, rules: Set[LocalRule], x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate Bayesian Information Criterion (BIC) for a set of rules.
        BIC = -2 * log_likelihood + complexity_penalty
        Lower is better.
        """
        if len(rules) == 0:
            return float('inf')
        
        # Calculate fidelity (log likelihood approximation)
        total_fidelity = 0.0
        total_coverage = 0
        for rule in rules:
            fidelity = self.binary_fidelity(rule, x, y)
            coverage = self.coverage(rule, x).sum()
            total_fidelity += fidelity * coverage
            total_coverage += coverage
        
        if total_coverage == 0:
            log_likelihood = 0.0
        else:
            avg_fidelity = total_fidelity / total_coverage if total_coverage > 0 else 0.0
            log_likelihood = np.log(avg_fidelity + 1e-10) * total_coverage
        
        # Complexity penalty: number of rules + total number of conditions
        total_conditions = sum(len(rule.features) for rule in rules)
        complexity = len(rules) * self.complexity_weight + total_conditions * 0.1
        
        bic_score = -2 * log_likelihood * self.fidelity_weight + complexity
        return bic_score
    
    def partition(self, A: List[LocalRule], B: List[LocalRule], 
                 x: np.ndarray, intersecting: str = 'coverage') -> Tuple[List[Set], List[Set], Set]:
        """
        Partition rules into conflicting, non-conflicting, and disjoint groups.
        
        Returns:
            conflicting_groups: List of sets of conflicting rules
            non_conflicting_groups: List of sets of non-conflicting rules
            disjoint: Set of disjoint rules
        """
        conflicting_groups = []
        non_conflicting_groups = []
        disjoint_A = set(A)
        disjoint_B = set(B)
        
        for a in A:
            conflicting_a = set()
            non_conflicting_a = set()
            coverage_a = self.coverage(a, x)
            
            for b in B:
                coverage_b = self.coverage(b, x)
                
                # Check intersection
                if intersecting == 'coverage':
                    a_intersecting_b = (coverage_a & coverage_b).any()
                else:  # polyhedra
                    intersection_rule = a & b
                    a_intersecting_b = intersection_rule is not None
                
                if a_intersecting_b:
                    if a.consequence != b.consequence:
                        # Conflicting
                        conflicting_a.add(a)
                        conflicting_a.add(b)
                        disjoint_A.discard(a)
                        disjoint_B.discard(b)
                    else:
                        # Non-conflicting (same consequence)
                        non_conflicting_a.add(a)
                        non_conflicting_a.add(b)
                        disjoint_A.discard(a)
                        disjoint_B.discard(b)
            
            if conflicting_a:
                conflicting_groups.append(conflicting_a)
            if non_conflicting_a:
                non_conflicting_groups.append(non_conflicting_a)
        
        disjoint = disjoint_A | disjoint_B
        return conflicting_groups, non_conflicting_groups, disjoint
    
    def _cut(self, conflicting_group: Set[LocalRule], x: np.ndarray, y: np.ndarray) -> Set[LocalRule]:
        """Cut conflicting rules, keeping the dominant one."""
        if len(conflicting_group) == 0:
            return conflicting_group
        
        conflicting_list = list(conflicting_group)
        default = int(y.mean().round())
        
        # Find dominant rule (highest fidelity)
        fidelities = np.array([self.binary_fidelity(rule, x, y) for rule in conflicting_list])
        dominant_rule = conflicting_list[np.argmax(fidelities)]
        
        cut_rules = {dominant_rule}
        
        # Cut other rules by removing features that conflict with dominant rule
        for rule in conflicting_group - {dominant_rule}:
            if self.strict_cut:
                # Remove all features that are in dominant rule
                remaining_features = rule.features - dominant_rule.features
            else:
                # Only remove shared conflicting features
                shared_features = rule.features & dominant_rule.features
                # Check which shared features conflict
                conflicting_features = set()
                for feature in shared_features:
                    lower1, upper1 = rule.premises[feature]
                    lower2, upper2 = dominant_rule.premises[feature]
                    # Check if ranges overlap
                    if not (lower1 < upper2 and lower2 < upper1):
                        conflicting_features.add(feature)
                remaining_features = rule.features - conflicting_features
            
            if len(remaining_features) > 0:
                new_premises = {f: rule.premises[f] for f in remaining_features}
                cut_rules.add(LocalRule(new_premises, rule.consequence, rule.rule_string))
        
        return cut_rules
    
    def _join(self, rules: Set[LocalRule], x: np.ndarray, y: np.ndarray) -> Set[LocalRule]:
        """Join non-conflicting rules with same consequence."""
        rules_list = list(rules)
        if len(rules_list) == 0:
            return rules
        
        # Find best rule (highest fidelity)
        default = int(y.mean().round())
        fidelities = np.array([self.binary_fidelity(rule, x, y) for rule in rules_list])
        best_rule = rules_list[np.argmax(fidelities)]
        
        # Collect ranges per feature
        ranges_per_feature = defaultdict(list)
        for rule in rules_list:
            for feature, bounds in rule.premises.items():
                ranges_per_feature[feature].append(bounds)
        
        # Features shared by all rules
        shared_features = {f: ranges_per_feature[f] 
                          for f in ranges_per_feature 
                          if len(ranges_per_feature[f]) == len(rules_list)}
        
        # Merge shared features: take intersection of ranges
        new_premises = {}
        for feature, ranges in shared_features.items():
            lower_bounds = [lower for lower, _ in ranges]
            upper_bounds = [upper for _, upper in ranges]
            new_lower = max(lower_bounds)
            new_upper = min(upper_bounds)
            if new_lower < new_upper:  # Valid intersection
                new_premises[feature] = (new_lower, new_upper)
        
        # If not strict_join, add non-shared features from best rule
        if not self.strict_join:
            for feature, bounds in best_rule.premises.items():
                if feature not in new_premises:
                    new_premises[feature] = bounds
        
        if len(new_premises) == 0:
            return {best_rule}
        
        merged_rule = LocalRule(new_premises, best_rule.consequence)
        return {merged_rule}
    
    def merge(self, A: Set[LocalRule], B: Set[LocalRule], 
             x: np.ndarray, y: np.ndarray) -> Set[LocalRule]:
        """Merge two sets of rules."""
        A_list = list(A)
        B_list = list(B)
        AB = set()
        
        # Partition rules
        conflicting_groups, non_conflicting_groups, disjoint = self.partition(
            A_list, B_list, x, intersecting='coverage'
        )
        
        # Add disjoint rules
        AB.update(disjoint)
        
        # Process conflicting groups (cut)
        for conflicting_group in conflicting_groups:
            cut_rules = self._cut(conflicting_group, x, y)
            AB.update(cut_rules)
        
        # Process non-conflicting groups (join)
        for non_conflicting_group in non_conflicting_groups:
            joined_rules = self._join(non_conflicting_group, x, y)
            AB.update(joined_rules)
        
        return AB
    
    def accept_merge(self, union: Set[LocalRule], merged: Set[LocalRule],
                    x: np.ndarray, y: np.ndarray) -> bool:
        """Decide whether to accept a merge based on BIC."""
        bic_union = self.bic(union, x, y)
        bic_merged = self.bic(merged, x, y)
        return bic_merged <= bic_union
    
    def aggregate(self, local_rules_list: List[Dict], x: np.ndarray, y: np.ndarray,
                 alpha: float = 0.5, max_iterations: int = 100) -> List[LocalRule]:
        """
        Aggregate local rules into global rules.
        
        Args:
            local_rules_list: List of local rule dictionaries from extract_local_rules
            x: Training data
            y: Training labels
            alpha: Pruning factor (0-1), lower means more aggressive pruning
            max_iterations: Maximum iterations for merging
        
        Returns:
            List of global rules
        """
        # Convert local rules to LocalRule objects
        all_rules = []
        for local_rule_dict in local_rules_list:
            clusters = local_rule_dict.get('clusters', {})
            for cluster_id, cluster_info in clusters.items():
                explanation_rule = cluster_info.get('explanation_rule')
                if explanation_rule:
                    parsed_rule = parse_rule_string(explanation_rule, self.feature_names)
                    if parsed_rule:
                        all_rules.append(parsed_rule)
        
        if len(all_rules) == 0:
            return []
        
        # Start with all rules as separate sets
        rule_sets = [{rule} for rule in all_rules]
        
        # Iteratively merge rules
        iteration = 0
        while iteration < max_iterations and len(rule_sets) > 1:
            iteration += 1
            merged_any = False
            
            # Try to merge pairs of rule sets
            new_rule_sets = []
            used = set()
            
            for i in range(len(rule_sets)):
                if i in used:
                    continue
                
                best_merge = None
                best_j = None
                best_merged_set = None
                
                for j in range(i + 1, len(rule_sets)):
                    if j in used:
                        continue
                    
                    union = rule_sets[i] | rule_sets[j]
                    merged = self.merge(rule_sets[i], rule_sets[j], x, y)
                    
                    if self.accept_merge(union, merged, x, y):
                        if best_merge is None or len(merged) < len(best_merged_set):
                            best_merge = merged
                            best_j = j
                            best_merged_set = merged
                
                if best_merge is not None:
                    new_rule_sets.append(best_merged_set)
                    used.add(i)
                    used.add(best_j)
                    merged_any = True
                else:
                    new_rule_sets.append(rule_sets[i])
                    used.add(i)
            
            if not merged_any:
                break
            
            rule_sets = new_rule_sets
        
        # Flatten and filter by alpha (fidelity threshold)
        global_rules = []
        for rule_set in rule_sets:
            for rule in rule_set:
                fidelity = self.binary_fidelity(rule, x, y)
                if fidelity >= alpha:
                    global_rules.append(rule)
        
        # Sort by fidelity (descending)
        global_rules.sort(key=lambda r: self.binary_fidelity(r, x, y), reverse=True)
        
        return global_rules


def visualize_global_rules(global_rules: List[LocalRule], aggregator: GlobalRuleAggregator,
                           x: np.ndarray, y: np.ndarray, feature_names: List[str],
                           top_k: int = 10):
    """
    Visualize global rules with their coverage, fidelity, and feature importance.
    
    Args:
        global_rules: List of global rules
        aggregator: GlobalRuleAggregator instance
        x: Training data
        y: Training labels
        feature_names: List of feature names
        top_k: Number of top rules to display
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if len(global_rules) == 0:
        print("No global rules to visualize.")
        return
    
    # Calculate metrics for each rule
    rule_metrics = []
    for i, rule in enumerate(global_rules[:top_k]):
        coverage = aggregator.coverage(rule, x)
        fidelity = aggregator.binary_fidelity(rule, x, y)
        coverage_size = coverage.sum()
        coverage_pct = (coverage_size / len(x)) * 100
        
        rule_metrics.append({
            'rule_id': i,
            'rule': str(rule),
            'fidelity': fidelity,
            'coverage': coverage_size,
            'coverage_pct': coverage_pct,
            'n_features': len(rule.features),
            'consequence': rule.consequence
        })
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Fidelity vs Coverage scatter
    ax1 = axes[0, 0]
    fidelities = [m['fidelity'] for m in rule_metrics]
    coverages = [m['coverage_pct'] for m in rule_metrics]
    colors = ['red' if m['consequence'] == 0 else 'green' for m in rule_metrics]
    ax1.scatter(coverages, fidelities, c=colors, alpha=0.6, s=100)
    ax1.set_xlabel('Coverage (%)', fontsize=12)
    ax1.set_ylabel('Fidelity', fontsize=12)
    ax1.set_title('Global Rules: Fidelity vs Coverage', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    for i, m in enumerate(rule_metrics):
        ax1.annotate(f"R{i}", (m['coverage_pct'], m['fidelity']), 
                    fontsize=8, alpha=0.7)
    
    # 2. Feature frequency in rules
    ax2 = axes[0, 1]
    feature_counts = defaultdict(int)
    for rule in global_rules[:top_k]:
        for feature in rule.features:
            feature_counts[feature] += 1
    
    if feature_counts:
        features_sorted = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        features, counts = zip(*features_sorted)
        ax2.barh(range(len(features)), counts)
        ax2.set_yticks(range(len(features)))
        ax2.set_yticklabels(features, fontsize=9)
        ax2.set_xlabel('Frequency in Global Rules', fontsize=12)
        ax2.set_title('Most Important Features in Global Rules', fontsize=14, fontweight='bold')
        ax2.invert_yaxis()
    
    # 3. Rule complexity (number of features)
    ax3 = axes[1, 0]
    complexities = [m['n_features'] for m in rule_metrics]
    ax3.hist(complexities, bins=min(10, max(complexities)), edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Number of Features per Rule', fontsize=12)
    ax3.set_ylabel('Number of Rules', fontsize=12)
    ax3.set_title('Distribution of Rule Complexity', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Coverage distribution
    ax4 = axes[1, 1]
    ax4.hist(coverages, bins=min(15, len(rule_metrics)), edgecolor='black', alpha=0.7, color='skyblue')
    ax4.set_xlabel('Coverage (%)', fontsize=12)
    ax4.set_ylabel('Number of Rules', fontsize=12)
    ax4.set_title('Distribution of Rule Coverage', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig, rule_metrics


def print_global_rules(global_rules: List[LocalRule], aggregator: GlobalRuleAggregator,
                      x: np.ndarray, y: np.ndarray, top_k: int = 10):
    """Print global rules in a human-readable format."""
    if len(global_rules) == 0:
        print("No global rules to display.")
        return
    
    print(f"\n{'='*80}")
    print(f"GLOBAL RULES (Top {min(top_k, len(global_rules))} of {len(global_rules)})")
    print(f"{'='*80}\n")
    
    for i, rule in enumerate(global_rules[:top_k]):
        coverage = aggregator.coverage(rule, x)
        fidelity = aggregator.binary_fidelity(rule, x, y)
        coverage_size = coverage.sum()
        coverage_pct = (coverage_size / len(x)) * 100
        
        print(f"Rule {i+1}:")
        print(f"  {str(rule)}")
        print(f"  Fidelity: {fidelity:.4f}")
        print(f"  Coverage: {coverage_size} samples ({coverage_pct:.2f}%)")
        print(f"  Features: {len(rule.features)}")
        print()

def visualize_global_rules_as_tree(global_rules: List[LocalRule], aggregator: GlobalRuleAggregator,
                                   x: np.ndarray, y: np.ndarray, feature_names: List[str],
                                   top_k: int = 10, max_depth: int = 8):
    """
    Visualize global rules directly as a decision tree structure.
    Each rule becomes a path from root to leaf in the tree.
    Does NOT re-train - directly visualizes the existing rules.
    
    Args:
        global_rules: List of global rules to visualize
        aggregator: GlobalRuleAggregator instance (for coverage/fidelity if needed)
        x: Training data (for statistics only)
        y: Training labels (for statistics only)
        feature_names: List of feature names
        top_k: Number of top rules to include
        max_depth: Maximum depth of the visualization
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    
    if len(global_rules) == 0:
        print("No global rules to visualize.")
        return None, None
    
    # Use top_k rules
    top_rules = global_rules[:top_k]
    
    # Build tree structure from rules
    # Each rule is a path: root -> feature1 -> feature2 -> ... -> leaf (consequence)
    rule_paths = []
    
    for rule_idx, rule in enumerate(top_rules):
        # Get rule statistics
        coverage = aggregator.coverage(rule, x) if aggregator else np.ones(len(x), dtype=bool)
        coverage_size = coverage.sum()
        coverage_pct = (coverage_size / len(x)) * 100 if len(x) > 0 else 0
        fidelity = aggregator.binary_fidelity(rule, x, y) if aggregator else 0.0
        
        # Build path for this rule
        path = []
        current_node = "root"
        path.append(current_node)
        
        # Sort features for consistent ordering
        sorted_features = sorted(rule.features)
        
        for feature in sorted_features[:max_depth-1]:  # Limit depth
            lower, upper = rule.premises[feature]
            
            # Create node ID
            if lower == float('-inf'):
                condition = f"{feature} <= {upper:.2f}"
            elif upper == float('inf'):
                condition = f"{feature} > {lower:.2f}"
            else:
                condition = f"{lower:.2f} < {feature} <= {upper:.2f}"
            
            node_id = f"{current_node}->{feature}"
            path.append(node_id)
            current_node = node_id
        
        # Leaf node (consequence)
        leaf_id = f"{current_node}->class{rule.consequence}"
        path.append(leaf_id)
        
        rule_paths.append({
            'rule_idx': rule_idx,
            'path': path,
            'rule': rule,
            'consequence': rule.consequence,
            'coverage': coverage_size,
            'coverage_pct': coverage_pct,
            'fidelity': fidelity
        })
    
    # Create visualization using matplotlib
    fig, ax = plt.subplots(figsize=(20, max(12, len(top_rules) * 0.8)))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(top_rules) + 1)
    ax.axis('off')
    
    # Draw tree structure
    y_pos = len(top_rules)
    node_positions = {}
    
    # Draw each rule as a horizontal path
    for path_info in rule_paths:
        rule_idx = path_info['rule_idx']
        rule = path_info['rule']
        path = path_info['path']
        consequence = path_info['consequence']
        coverage_pct = path_info['coverage_pct']
        fidelity = path_info['fidelity']
        
        y = y_pos - rule_idx - 0.5
        
        # Draw path from left to right
        x_start = 0.5
        x_step = 1.5
        x_pos = x_start
        
        # Root node
        root_box = FancyBboxPatch((x_pos - 0.2, y - 0.15), 0.4, 0.3,
                                 boxstyle="round,pad=0.05", 
                                 facecolor='lightblue', edgecolor='black', linewidth=1.5)
        ax.add_patch(root_box)
        ax.text(x_pos, y, 'ROOT', ha='center', va='center', fontsize=8, fontweight='bold')
        node_positions[path[0]] = (x_pos, y)
        x_pos += x_step
        
        # Intermediate nodes (features)
        for i, node_id in enumerate(path[1:-1], 1):
            feature = sorted(rule.features)[i-1] if i-1 < len(sorted(rule.features)) else "?"
            lower, upper = rule.premises.get(feature, (0, 0))
            
            if lower == float('-inf'):
                label = f"{feature}\n≤ {upper:.2f}"
            elif upper == float('inf'):
                label = f"{feature}\n> {lower:.2f}"
            else:
                label = f"{lower:.2f} <\n{feature}\n≤ {upper:.2f}"
            
            # Node box
            node_box = FancyBboxPatch((x_pos - 0.25, y - 0.2), 0.5, 0.4,
                                     boxstyle="round,pad=0.05",
                                     facecolor='lightyellow', edgecolor='black', linewidth=1.5)
            ax.add_patch(node_box)
            ax.text(x_pos, y, label, ha='center', va='center', fontsize=7)
            node_positions[node_id] = (x_pos, y)
            
            # Arrow from previous node
            prev_x, prev_y = node_positions[path[i-1]]
            arrow = FancyArrowPatch((prev_x + 0.2, prev_y), (x_pos - 0.25, y),
                                   arrowstyle='->', mutation_scale=20, 
                                   color='gray', linewidth=1.5)
            ax.add_patch(arrow)
            
            x_pos += x_step
        
        # Leaf node (consequence)
        leaf_color = 'lightcoral' if consequence == 0 else 'lightgreen'
        leaf_box = FancyBboxPatch((x_pos - 0.25, y - 0.2), 0.5, 0.4,
                                 boxstyle="round,pad=0.05",
                                 facecolor=leaf_color, edgecolor='black', linewidth=2)
        ax.add_patch(leaf_box)
        leaf_label = f"Class {consequence}\n({coverage_pct:.1f}%)\nF:{fidelity:.2f}"
        ax.text(x_pos, y, leaf_label, ha='center', va='center', 
               fontsize=8, fontweight='bold')
        node_positions[path[-1]] = (x_pos, y)
        
        # Arrow to leaf
        if len(path) > 2:
            prev_x, prev_y = node_positions[path[-2]]
            arrow = FancyArrowPatch((prev_x + 0.25, prev_y), (x_pos - 0.25, y),
                                   arrowstyle='->', mutation_scale=20,
                                   color='gray', linewidth=1.5)
            ax.add_patch(arrow)
        
        # Rule label on the left
        ax.text(0.1, y, f"R{rule_idx+1}", ha='left', va='center', 
               fontsize=10, fontweight='bold', 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Title and legend
    ax.text(5, len(top_rules) + 0.5, 
           f"Global Rules as Decision Tree (Top {len(top_rules)} Rules)",
           ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='lightblue', label='Root'),
        mpatches.Patch(facecolor='lightyellow', label='Feature Condition'),
        mpatches.Patch(facecolor='lightcoral', label='Class 0 (Reject)'),
        mpatches.Patch(facecolor='lightgreen', label='Class 1 (Approve)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Global Rules Tree Visualization")
    print(f"{'='*70}")
    print(f"Total rules visualized: {len(top_rules)}")
    print(f"Each path represents one global rule")
    print(f"Leaf nodes show: Class prediction, Coverage %, Fidelity")
    print(f"{'='*70}\n")
    
    return fig, rule_paths


def visualize_global_rules_hierarchy(global_rules: List[LocalRule], aggregator: GlobalRuleAggregator,
                                    x: np.ndarray, y: np.ndarray, feature_names: List[str],
                                    top_k: int = 10):
    """
    Visualize global rules as a hierarchical tree showing rule relationships.
    Each node represents a rule, and edges show feature overlaps.
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.patches import FancyBboxPatch
    
    if len(global_rules) == 0:
        print("No global rules to visualize.")
        return None
    
    top_rules = global_rules[:top_k]
    
    # Create a graph where nodes are rules and edges represent feature overlap
    G = nx.Graph()
    
    # Add nodes (rules)
    for i, rule in enumerate(top_rules):
        fidelity = aggregator.binary_fidelity(rule, x, y)
        coverage = aggregator.coverage(rule, x)
        coverage_pct = (coverage.sum() / len(x)) * 100
        
        G.add_node(i, 
                  rule=rule,
                  fidelity=fidelity,
                  coverage_pct=coverage_pct,
                  n_features=len(rule.features),
                  consequence=rule.consequence)
    
    # Add edges based on feature overlap
    for i in range(len(top_rules)):
        for j in range(i + 1, len(top_rules)):
            overlap = len(top_rules[i].features & top_rules[j].features)
            if overlap > 0:
                # Weight edge by overlap
                G.add_edge(i, j, weight=overlap)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Use spring layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Draw nodes with color based on consequence
    node_colors = ['lightcoral' if G.nodes[i]['consequence'] == 0 else 'lightgreen' 
                   for i in G.nodes()]
    node_sizes = [G.nodes[i]['coverage_pct'] * 50 for i in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=node_sizes, alpha=0.7, ax=ax)
    
    # Draw edges with thickness based on overlap
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, width=[w * 0.5 for w in weights], 
                            alpha=0.3, edge_color='gray', ax=ax)
    
    # Add labels
    labels = {}
    for i in G.nodes():
        rule = G.nodes[i]['rule']
        # Create short label
        features_str = ', '.join(list(rule.features)[:3])
        if len(rule.features) > 3:
            features_str += '...'
        labels[i] = f"R{i}\nClass {rule.consequence}\n{len(rule.features)} features"
    
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightcoral', label='Class 0 (Reject)'),
        Patch(facecolor='lightgreen', label='Class 1 (Approve)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_title(f"Global Rules Hierarchy (Top {len(top_rules)} Rules)\n"
                f"Node size = Coverage %, Edge thickness = Feature overlap",
                fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    # Print rule relationships
    print(f"\nRule Relationships:")
    print(f"  Total rules: {len(top_rules)}")
    print(f"  Total edges (overlaps): {len(G.edges())}")
    print(f"  Average features per rule: {np.mean([G.nodes[i]['n_features'] for i in G.nodes()]):.1f}")
    
    return fig, G
