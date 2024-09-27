import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict

class Node:
    def __init__(self, level=0):
        self.centroid = None
        self.children = []
        self.level = level
        self.visual_word_id = None  # For leaf nodes
        self.df = 0                 # Document frequency
        self.posting_list = defaultdict(int)  # Image ID to term frequency

def hi_kmeans(data, labels, b, depth):
    """
    Builds a vocabulary tree using hierarchical k-means clustering.
    Parameters:
    - data (numpy array): SIFT features of shape {(number of key points * number of images) * 128 features}.
    - labels (numpy array): one dimensional labels concatenate for each key points for all images.
    - b (int): Branching factor (number of clusters at each node).
    - depth (int): Depth of the tree.

    Returns:
    - root (Node): Root node of the vocabulary tree.
    """
    visual_word_id_counter = [0]  # Mutable counter for assigning visual word IDs

    def recursive_kmeans(data_indices, current_depth):
        """
            recursively fill in k-means results.
            Parameters:
            - data_indices (numpy array): Arrange of indices of data points.
            - current_depth (int): Depth of the tree, start from 0.

            Return:
            - node (Node): K-means node of the vocabulary tree.
            """
        node = Node(level=current_depth)
        # list of all key points
        data_points = data[data_indices]

        # Compute centroid
        node.centroid = np.mean(data_points, axis=0)

        if current_depth < depth:
            # Determine the number of clusters
            num_samples = len(data_points)
            num_clusters = min(b, num_samples)
            if num_clusters > 1:
                # Perform k-means clustering
                kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10)
                kmeans.fit(data_points)
                labels_kmeans = kmeans.labels_

                # Process child clusters
                for i in range(num_clusters):
                    child_indices = data_indices[labels_kmeans == i]
                    if len(child_indices) > 0:
                        # do hierarchical kmeans under
                        child_node = recursive_kmeans(child_indices, current_depth + 1)
                        node.children.append(child_node)
            else:
                # Not enough data points to split further, make this a leaf node
                visual_word_id_counter[0] += 1
                node.visual_word_id = visual_word_id_counter[0]

                # Compute posting list and document frequency
                leaf_labels = labels[data_indices]
                unique_labels, counts = np.unique(leaf_labels, return_counts=True)
                node.posting_list = dict(zip(unique_labels, counts))
                node.df = len(unique_labels)
        else:
            # Leaf node
            visual_word_id_counter[0] += 1
            node.visual_word_id = visual_word_id_counter[0]

            # Compute posting list and document frequency
            leaf_labels = labels[data_indices]
            unique_labels, counts = np.unique(leaf_labels, return_counts=True)
            node.posting_list = dict(zip(unique_labels, counts))
            node.df = len(unique_labels)

        return node

    # Start recursion from the root
    indices = np.arange(len(data))
    root = recursive_kmeans(indices, current_depth=0)
    return root, visual_word_id_counter[0]  # Return root and total number of visual words