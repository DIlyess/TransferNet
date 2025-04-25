import random
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

# from node2vec import Node2Vec
from scipy.stats import randint, uniform
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class TransferGraphDataProcessor:
    """
    Class for processing football transfer graphs, including splitting data into train and test sets
    by artificially removing edges and preparing features for edge prediction.
    """

    def __init__(self, graph, test_size=0.2, random_state=42):
        """
        Initialize the processor with a directed graph of football transfers.

        Parameters:
            - graph (networkx.DiGraph) : representing football transfers
            - test_size (float) : Proportion of edges to remove for testing (default: 0.2)
            - random_state : int
        """
        self.original_graph = graph.copy()
        self.train_graph = None
        self.test_edges = None
        self.test_non_edges = None
        self.test_size = test_size
        self.random_state = random_state
        self.node_features = None
        self.feature_names = None
        self.scaler = StandardScaler()
        random.seed(random_state)
        np.random.seed(random_state)

    def split_edges(self, balance_test_set=True):
        """
        Split the graph into training and test sets by removing edges.

        Parameters:
            - balance_test_set (bool): Whether to create balanced test set with equal positive and negative examples

        Returns:
            - train_graph (nx.Digraph) : Graph with edges removed for testing
            - test_edges : List of edges removed for testing
            - test_non_edges : List of non-existent edges for testing negative examples
        """
        all_edges = list(self.original_graph.edges(data=True))

        # Split edges into train and test sets
        train_edges, test_edges = train_test_split(
            all_edges, test_size=self.test_size, random_state=self.random_state
        )

        # Create training graph by removing test edges
        self.train_graph = self.original_graph.copy()
        self.test_edges = []

        for u, v, attrs in test_edges:
            if self.train_graph.has_edge(u, v):
                # Store the edge with attributes for later evaluation
                self.test_edges.append((u, v, attrs))
                self.train_graph.remove_edge(u, v)

        # Create negative examples (non-edges)
        if balance_test_set:
            self.test_non_edges = self._generate_negative_examples(
                self.original_graph, len(self.test_edges)
            )
        else:
            # Generate a fixed number of negative examples
            self.test_non_edges = self._generate_negative_examples(
                self.original_graph, len(self.test_edges) * 2
            )

        return self.train_graph, self.test_edges, self.test_non_edges

    def _generate_negative_examples(self, graph, num_samples):
        """
        Generate negative examples (non-edges) in a graph with random attributes.

        Parameters:
            - graph : (nx.DiGraph) graph to generate non-edges from
            - num_samples (int) : nb of negative examples to generate
        Returns:
            - non_edges (list) : List of tuples (u, v, random_attrs) representing non-existent edges
        """
        nodes = list(graph.nodes())
        non_edges = []

        # Set a limit to avoid infinite loops
        max_attempts = num_samples * 10
        attempts = 0

        while len(non_edges) < num_samples and attempts < max_attempts:
            u = random.choice(nodes)
            v = random.choice(nodes)

            # Skip self-loops and existing edges
            if u != v and not graph.has_edge(u, v):
                # create random attributes for the non-edge
                # random_attrs = {}
                # for key, (min_val, max_val) in attribute_ranges.items():
                #     random_attrs[key] = random.uniform(min_val, max_val)
                non_edges.append((u, v, None))

            attempts += 1

        return non_edges

    def extract_node_features(self, method="structural"):
        """
        Extract node features from the graph.

        Parameters:
            - method (str) : Method to extract features ('structural', 'node2vec', 'both')

        Returns:
            None
        """
        features = {}

        if method in ["structural", "both"]:
            # Compute structural features
            in_degree = dict(self.train_graph.in_degree())
            out_degree = dict(self.train_graph.out_degree())

            for node, node_attr in self.train_graph.nodes(data=True):
                features[node] = {
                    "in_degree": in_degree.get(node, 0),
                    "out_degree": out_degree.get(node, 0),
                }

                features[node]["country"] = node_attr.get("country", "Unknown")

                # Additional structural features
                neighbors = list(self.train_graph.neighbors(node))
                predecessors = list(self.train_graph.predecessors(node))

                features[node]["clustering"] = nx.clustering(
                    self.train_graph.to_undirected(), node
                )
                features[node]["neighbors_count"] = len(set(neighbors + predecessors))

                # Transfer value statistics for outgoing and incoming transfers
                out_transfers = [
                    d.get("total_fee", 0)
                    for _, _, d in self.train_graph.out_edges(node, data=True)
                ]
                in_transfers = [
                    d.get("total_fee", 0)
                    for _, _, d in self.train_graph.in_edges(node, data=True)
                ]

                features[node]["avg_outgoing_fee"] = (
                    np.mean(out_transfers) if out_transfers else 0
                )
                features[node]["avg_incoming_fee"] = (
                    np.mean(in_transfers) if in_transfers else 0
                )
                features[node]["loan_out_ratio"] = sum(
                    1
                    for _, _, d in self.train_graph.out_edges(node, data=True)
                    if d.get("is_loan", False)
                ) / max(1, len(out_transfers))
                features[node]["loan_in_ratio"] = sum(
                    1
                    for _, _, d in self.train_graph.in_edges(node, data=True)
                    if d.get("is_loan", False)
                ) / max(1, len(in_transfers))

        # if method in ["node2vec", "both"]:
        #     # Generate node2vec embeddings
        #     try:
        #         node2vec = Node2Vec(
        #             self.train_graph,
        #             dimensions=64,
        #             walk_length=30,
        #             num_walks=200,
        #             workers=4,
        #         )
        #         model = node2vec.fit(window=10, min_count=1)

        #         # Add embeddings to features
        #         for node in self.train_graph.nodes():
        #             if node in model.wv:
        #                 if method == "node2vec":
        #                     features[node] = {
        #                         f"emb_{i}": val for i, val in enumerate(model.wv[node])
        #                     }
        #                 else:  # method == 'both'
        #                     for i, val in enumerate(model.wv[node]):
        #                         features[node][f"emb_{i}"] = val
        #             else:
        #                 # Handle nodes not in embeddings
        #                 if method == "node2vec":
        #                     features[node] = {f"emb_{i}": 0.0 for i in range(64)}
        #                 else:  # method == 'both'
        #                     for i in range(64):
        #                         features[node][f"emb_{i}"] = 0.0
        #     except Exception as e:
        #         print(f"Warning: Could not generate node2vec embeddings. Error: {e}")
        #         print("Falling back to structural features only.")

        self.node_features = features

    def _extract_edge_features(self, u, v, graph):
        """
        Extract features for an edge (u, v) based on node features and graph properties.

        Parameters:
            - u, v : node ids forming the edge
            - graph : Graph to extract additional features from

        Returns:
            - features (dict) : Feature dictionary for the edge
        """

        if self.node_features is None:
            raise ValueError(
                "Node features must be extracted first using extract_node_features()"
            )

        # Get node features
        u_features = self.node_features.get(u, {})
        v_features = self.node_features.get(v, {})

        # Basic node feature combinations
        features = {}
        features["same_country"] = int(
            u_features.get("country") == v_features.get("country")
        )
        # Iterate through all features and create combinations
        for key in set(u_features.keys()) | set(v_features.keys()):
            u_val = u_features.get(key, 0)
            v_val = v_features.get(key, 0)
            if isinstance(u_val, (int, float)) and isinstance(v_val, (int, float)):
                features[f"u_{key}"] = u_val
                features[f"v_{key}"] = v_val
                features[f"diff_{key}"] = u_val - v_val
                features[f"sum_{key}"] = u_val + v_val
                features[f"product_{key}"] = u_val * v_val

        # Graph-based edge features
        # Ensure nodes exist in the graph
        if u in graph.nodes() and v in graph.nodes():
            # Common neighbors in undirected sense
            u_neighbors = set(graph.successors(u)) | set(graph.predecessors(u))
            v_neighbors = set(graph.successors(v)) | set(graph.predecessors(v))
            common_neighbors = u_neighbors & v_neighbors

            features["common_neighbors"] = len(common_neighbors)

            # Jaccard coefficient
            try:
                features["jaccard"] = (
                    len(common_neighbors) / len(u_neighbors | v_neighbors)
                    if u_neighbors or v_neighbors
                    else 0
                )
            except ZeroDivisionError:
                features["jaccard"] = 0

            # Preferential attachment
            features["preferential_attachment"] = len(u_neighbors) * len(v_neighbors)

            # Resource allocation index
            undirected_graph = graph.to_undirected()
            try:
                features["resource_allocation"] = sum(
                    1 / len(list(undirected_graph.neighbors(w)))
                    for w in common_neighbors
                    if len(list(undirected_graph.neighbors(w))) > 0
                )
            except ZeroDivisionError:
                features["resource_allocation"] = 0

        return features

    def prepare_training_data(self, max_train_samples=10000):
        """
        Prepare training and testing data for edge prediction.

        Parameters:
            - max_train_samples (int): Maximum number of training samples to use

        Returns:
            - X_train, y_train, X_test, y_test (np.array) : Training and testing data
            - feature_names (list) : List of feature names
        """
        if (
            self.train_graph is None
            or self.test_edges is None
            or self.test_non_edges is None
            or self.node_features is None
        ):
            raise ValueError(
                "Must call split_edges() and then extract_node_features() before preparing training data"
            )

        # Prepare positive examples from training graph
        train_pos_edges = list(self.train_graph.edges())
        random.shuffle(train_pos_edges)

        # Limit the number of positive examples to avoid imbalance
        max_examples = min(max_train_samples, len(train_pos_edges))
        train_pos_edges = train_pos_edges[:max_examples]

        # Generate negative examples for training (ensure we really do generate negatives)
        print("Generating negative examples for training...")

        train_neg_edges = self._generate_negative_examples(
            self.train_graph, len(train_pos_edges)
        )

        print(f"Train w/ {len(train_pos_edges)} positive, {len(train_neg_edges)} neg")

        # Create feature vectors for training data
        train_data = []

        print("Preparing training data...")
        for u, v in tqdm(train_pos_edges):
            features = self._extract_edge_features(u, v, self.train_graph)
            train_data.append((features, 1))  # Positive example

        for u, v, _ in tqdm(train_neg_edges):
            features = self._extract_edge_features(u, v, self.train_graph)
            train_data.append((features, 0))  # Negative example

        # Create feature vectors for test data
        test_data = []

        print("Preparing test data...")
        for u, v, _ in tqdm(self.test_edges):
            features = self._extract_edge_features(u, v, self.train_graph)
            test_data.append((features, 1))  # Positive example

        for u, v, _ in tqdm(self.test_non_edges):
            features = self._extract_edge_features(u, v, self.train_graph)
            test_data.append((features, 0))  # Negative example

        # Extract feature names from the first sample
        feature_dict = train_data[0][0]
        self.feature_names = sorted(feature_dict.keys())

        # Convert dictionaries to arrays
        X_train = np.array(
            [[sample[0].get(f, 0) for f in self.feature_names] for sample in train_data]
        )
        X_test = np.array(
            [[sample[0].get(f, 0) for f in self.feature_names] for sample in test_data]
        )

        y_train = np.array([sample[1] for sample in train_data])
        y_test = np.array([sample[1] for sample in test_data])

        # Handle any NaN values by replacing with 0
        if np.isnan(X_train).any():
            print("Warning: NaN values found in training data, replacing with 0")
            X_train = np.nan_to_num(X_train)
            X_test = np.nan_to_num(X_test)

        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, y_train, X_test, y_test, self.feature_names

    def get_edge_features_for_prediction(self, u, v):
        """
        Get scaled features for a specific edge for prediction.
        Parameters:
            - u, v : node ids forming the edge
        Returns:
            - features (np.array) : Scaled feature vector for the edge
        """
        if self.feature_names is None:
            raise ValueError(
                "Must call prepare_training_data() before getting edge features"
            )

        features_dict = self._extract_edge_features(u, v, self.train_graph)
        features = np.array([[features_dict.get(f, 0) for f in self.feature_names]])

        # Handle any NaN values
        features = np.nan_to_num(features)

        return self.scaler.transform(features)

    def sample_node_pairs_for_prediction(self, sample_size=10000):
        """
        Sample node pairs that don't have an edge for prediction.

        Parameters:
            - sample_size (int): Number of node pairs to sample

        Returns:
            - node_pairs : List of tuples (u, v) for node pairs without an edge
        """
        nodes = list(self.train_graph.nodes())
        n_nodes = len(nodes)

        if n_nodes < 2:
            raise ValueError("Graph has fewer than 2 nodes, cannot sample pairs")

        # Limit the number of pairs to check
        max_pairs = min(sample_size, n_nodes * (n_nodes - 1))
        node_pairs = []

        # Set a limit to avoid infinite loops
        max_attempts = max_pairs * 10
        attempts = 0

        while len(node_pairs) < max_pairs and attempts < max_attempts:
            u = random.choice(nodes)
            v = random.choice(nodes)

            if u != v and not self.train_graph.has_edge(u, v):
                node_pairs.append((u, v))

            attempts += 1

        return node_pairs


class TransferEdgePrediction:
    """
    Class for training models and performing edge prediction on football transfer graphs.
    """

    def __init__(self):
        """
        Initialize the edge prediction class.
        """
        self.models = {}
        self.feature_names = None

    def train_models(self, X_train, y_train, feature_names=None):
        """
        Train multiple models for edge prediction using wide hyperparameter tuning.

        Parameters:
            - X_train : Training features
            - y_train : Training labels
            - feature_names : List of feature names (optional)

        Returns:
            - self
        """
        print("Training models with hyperparameter tuning...")
        self.feature_names = feature_names
        self.models = {}

        scoring = make_scorer(f1_score, average="binary")

        # Logistic Regression - wide tuning
        lr_params = {
            "C": uniform(0.01, 100),
        }
        lr = RandomizedSearchCV(
            LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
            param_distributions=lr_params,
            n_iter=20,
            scoring=scoring,
            cv=5,
            random_state=42,
            n_jobs=-1,
        )
        lr.fit(X_train, y_train)
        self.models["logistic_regression"] = lr.best_estimator_

        # Random Forest - wide tuning
        rf_params = {
            "n_estimators": randint(50, 300),
            "max_depth": [None] + list(range(5, 30)),
            "min_samples_split": randint(2, 10),
            "min_samples_leaf": randint(1, 10),
            "max_features": ["sqrt", "log2", None],
        }
        rf = RandomizedSearchCV(
            RandomForestClassifier(random_state=42, class_weight="balanced"),
            param_distributions=rf_params,
            n_iter=20,
            scoring=scoring,
            cv=5,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        self.models["random_forest"] = rf.best_estimator_

        # Gradient Boosting - wide tuning
        gb_params = {
            "n_estimators": randint(50, 300),
            "learning_rate": uniform(0.01, 0.3),
            "max_depth": randint(3, 10),
            "subsample": uniform(0.5, 0.5),
            "min_samples_split": randint(2, 10),
            "min_samples_leaf": randint(1, 10),
            "max_features": ["sqrt", "log2", None],
        }
        gb = RandomizedSearchCV(
            GradientBoostingClassifier(random_state=42),
            param_distributions=gb_params,
            n_iter=20,
            scoring=scoring,
            cv=5,
            random_state=42,
            n_jobs=-1,
        )
        gb.fit(X_train, y_train)
        self.models["gradient_boosting"] = gb.best_estimator_

        return self

    def evaluate_models(self, X_test, y_test):
        """
        Evaluate trained models.
        Parameters:
            - X_test : Test features
            - y_test : Test labels
        Returns:
            - results : Dictionary of evaluation metrics
        """
        results = {}

        # Verify we have both classes in the test data
        unique_classes = np.unique(y_test)
        if len(unique_classes) < 2:
            raise ValueError(
                f"Test data only contains class {unique_classes[0]}. Need both 0 and 1 for evaluation."
            )

        for model_name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            results[model_name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_prob),
            }

        return results

    def predict_edge_probability(self, X, model_name="gradient_boosting"):
        """
        Predict the probability of an edge for given features.

        Parameters:
            - X (np.array): Edge features (scaled)
            - model_name : Name of the model to use for prediction

        Returns:
            - probability : float Probability of edge existence
        """
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not found")

        return model.predict_proba(X)[0, 1]

    def predict_top_edges(
        self, node_pairs, processor, top_k=10, model_name="gradient_boosting"
    ):
        """
        Predict top-k most likely edges from a list of node pairs.

        Parameters:
            - node_pairs : list List of tuples (u, v) for node pairs to evaluate
            - processor : TransferGraphDataProcessor Data processor with scaling capabilities
            - top_k : int Number of top edges to return
            - model_name : str Name of the model to use for prediction

        Returns:
            - top_edges : List of tuples (u, v, probability) for top predicted edges
        """
        # Predict probability for each pair
        predictions = []

        print("Predicting edge probabilities...")
        for u, v in tqdm(node_pairs):
            try:
                X = processor.get_edge_features_for_prediction(u, v)
                prob = self.predict_edge_probability(X, model_name)
                predictions.append((u, v, prob))
            except Exception as e:
                print(f"Warning: Could not predict for edge ({u}, {v}). Error: {e}")

        if not predictions:
            print("No valid predictions made.")
            return []

        # Sort by probability and return top-k
        predictions.sort(key=lambda x: x[2], reverse=True)
        return predictions[:top_k]

    def plot_feature_importance(self, model_name="random_forest"):
        """
        Plot feature importance for tree-based models.

        Parameters:
            - model_name : Name of the model to use (must be tree-based)
        """

        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not found")

        # Get feature importance
        if model_name == "logistic_regression":
            importances = model.coef_[0]
        else:
            importances = model.feature_importances_

        # Sort features by importance
        sorted_idx = importances.argsort()

        # Plot top 20 features (or fewer if there are less features)
        top_n = len(sorted_idx)
        plt.figure(figsize=(12, 8))
        plt.barh(range(top_n), importances[sorted_idx][-top_n:])
        plt.yticks(range(top_n), [self.feature_names[i] for i in sorted_idx][-top_n:])
        plt.xlabel("Feature Importance")
        plt.title(f"Top {top_n} Feature Importance - {model_name}")
        plt.tight_layout()
        plt.show()
