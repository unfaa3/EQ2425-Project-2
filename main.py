from vocabulary_tree import hi_kmeans
import numpy as np
import math
from collections import defaultdict
import os

def traverse_tree(feature, node):
    """
    Traverses the tree to assign a visual word ID to a feature.

    Parameters:
    - feature (numpy array): The SIFT feature.
    - node (Node): Current node in the tree.

    Returns:
    - visual_word_id (int): The ID of the leaf node (visual word).
    """
    if not node.children:
        return node.visual_word_id
    else:
        # Compute distances to child centroids
        distances = [np.linalg.norm(feature - child.centroid) for child in node.children]
        # Find the child with the minimum distance
        min_index = np.argmin(distances)
        return traverse_tree(feature, node.children[min_index])

def get_visual_words(features, root):
    """
    Assigns visual word IDs to all features of an image.

    Parameters:
    - features (numpy array): SIFT features of the image.
    - root (Node): Root of the vocabulary tree.

    Returns:
    - visual_words (list): List of visual word IDs.
    """
    visual_words = []
    for feature in features:
        vw_id = traverse_tree(feature, root)
        visual_words.append(vw_id)
    return visual_words

# --- Compute TF-IDF Scores ---

def compute_idf(total_docs, visual_word_nodes):
    """
    Computes IDF values for all visual words.

    Parameters:
    - total_docs (int): Total number of documents (images).
    - visual_word_nodes (list of Node): List of leaf nodes.

    Returns:
    - idf (dict): Visual word ID to IDF value.
    """
    idf = {}
    for node in visual_word_nodes:
        df = node.df
        idf_value = math.log((total_docs) / (df + 1))  # Added 1 to avoid division by zero
        idf[node.visual_word_id] = idf_value
    return idf

def build_tfidf_vectors(image_visual_words, idf, num_visual_words):
    """
    Builds TF-IDF vectors for images.

    Parameters:
    - image_visual_words (dict): Image ID to list of visual word IDs.
    - idf (dict): Visual word ID to IDF value.
    - num_visual_words (int): Total number of visual words.

    Returns:
    - tfidf_vectors (dict): Image ID to TF-IDF vector (numpy array).
    """
    tfidf_vectors = {}
    for image_id, vw_list in image_visual_words.items():
        tf = defaultdict(int)
        for vw in vw_list:
            tf[vw] += 1
        # Build TF-IDF vector
        tfidf_vector = np.zeros(num_visual_words)
        for vw_id, count in tf.items():
            tfidf_vector[vw_id - 1] = count * idf.get(vw_id, 0)
        # Normalize the vector
        norm = np.linalg.norm(tfidf_vector)
        if norm > 0:
            tfidf_vector = tfidf_vector / norm
        tfidf_vectors[image_id] = tfidf_vector
    return tfidf_vectors


def rank_images(query_vector, db_tfidf_vectors):
    """
    Ranks database images based on similarity to the query vector.

    Parameters:
    - query_vector (numpy array): TF-IDF vector of the query image.
    - db_tfidf_vectors (dict): Image ID to TF-IDF vector.

    Returns:
    - ranked_list (list): List of tuples (image_id, similarity_score) sorted by score.
    """
    similarities = {}
    for image_id, db_vector in db_tfidf_vectors.items():
        sim = np.dot(query_vector, db_vector)
        similarities[image_id] = sim
    # Sort images by similarity score in descending order
    ranked_list = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return ranked_list


def part4_a():
    # Parameters
    b_values = [4, 4, 5]
    depth_values = [3, 5, 7]
    num_objects = 50

    # Directories
    database_dir = 'Data2/server'
    query_dir = 'Data2/client'

    # Load database features and labels
    db_features = []
    db_labels = []
    for obj_num in range(1, num_objects + 1):
        # Load combined descriptors for each object
        descriptors_path = os.path.join(database_dir, f'obj{obj_num}_descriptors.npy')
        descriptors = np.load(descriptors_path)
        db_features.append(descriptors)
        # Assign image ID to each feature
        labels = np.full(descriptors.shape[0], obj_num)
        db_labels.append(labels)
    db_features = np.vstack(db_features)
    db_labels = np.concatenate(db_labels)

    # Load query features
    query_features_dict = {}
    for obj_num in range(1, num_objects + 1):
        descriptors_path = os.path.join(query_dir, f'obj{obj_num}_descriptors.npy')
        descriptors = np.load(descriptors_path)
        query_features_dict[obj_num] = descriptors

    # Process for each set of parameters
    for b, depth in zip(b_values, depth_values):
        print(f'\n--- Vocabulary Tree with b = {b}, depth = {depth} ---\n')

        # db_features: {(number of key points) * 128 features}  vertical stack on each image
        # db_labels: one dimensional labels concatenate for each key points for each images
        root, total_visual_words = hi_kmeans(db_features, db_labels, b, depth)

        # Collect all leaf nodes (visual words)
        visual_word_nodes = []

        def collect_leaf_nodes(node):
            if not node.children:
                visual_word_nodes.append(node)
            else:
                for child in node.children:
                    collect_leaf_nodes(child)

        collect_leaf_nodes(root)

        # Assign visual words to database images
        image_visual_words = defaultdict(list)
        for obj_num in range(1, num_objects + 1):
            descriptors_path = os.path.join(database_dir, f'obj{obj_num}_descriptors.npy')
            descriptors = np.load(descriptors_path)
            visual_words = get_visual_words(descriptors, root)
            image_visual_words[obj_num].extend(visual_words)

        # Compute IDF values
        idf = compute_idf(num_objects, visual_word_nodes)

        # Build TF-IDF vectors for database images
        db_tfidf_vectors = build_tfidf_vectors(image_visual_words, idf, total_visual_words)

        # Process query images and rank database objects
        top1_correct = 0
        top5_correct = 0
        for obj_num in range(1, num_objects + 1):
            query_descriptors = query_features_dict[obj_num]
            query_visual_words = get_visual_words(query_descriptors, root)
            query_visual_words_dict = {obj_num: query_visual_words}
            query_tfidf_vectors = build_tfidf_vectors(query_visual_words_dict, idf, total_visual_words)
            query_vector = query_tfidf_vectors[obj_num]
            ranked_list = rank_images(query_vector, db_tfidf_vectors)
            top_results = [img_id for img_id, score in ranked_list[:5]]

            # Check if the correct object is in top-1 and top-5
            if top_results[0] == obj_num:
                top1_correct += 1
            if obj_num in top_results:
                top5_correct += 1

            print(f'Query Object {obj_num}: Top-5 Results: {top_results}')

        # Calculate recall rates
        top1_recall = top1_correct / num_objects
        top5_recall = top5_correct / num_objects

        print(f'\nResults for b = {b}, depth = {depth}:')
        print(f'Top-1 Recall Rate: {top1_recall * 100:.2f}%')
        print(f'Top-5 Recall Rate: {top5_recall * 100:.2f}%')


def part4_b():
    # Parameters
    b = 5
    depth = 7
    num_objects = 50
    percentages = [90, 70, 50]  # Percentages of query features to use

    # Directories
    database_dir = 'Data2/server'
    query_dir = 'Data2/client'

    # Load database features and labels
    db_features = []
    db_labels = []
    for obj_num in range(1, num_objects + 1):
        # Load combined descriptors for each object
        descriptors_path = os.path.join(database_dir, f'obj{obj_num}_descriptors.npy')
        descriptors = np.load(descriptors_path)
        db_features.append(descriptors)
        # Assign image ID to each feature
        labels = np.full(descriptors.shape[0], obj_num)
        db_labels.append(labels)
    db_features = np.vstack(db_features)
    db_labels = np.concatenate(db_labels)

    # Load query features
    query_features_dict = {}
    for obj_num in range(1, num_objects + 1):
        descriptors_path = os.path.join(query_dir, f'obj{obj_num}_descriptors.npy')
        descriptors = np.load(descriptors_path)
        query_features_dict[obj_num] = descriptors

    # Build the Vocabulary Tree once
    print(f'\n--- Building Vocabulary Tree with b = {b}, depth = {depth} ---\n')
    root, total_visual_words = hi_kmeans(db_features, db_labels, b, depth)

    # Collect all leaf nodes (visual words)
    visual_word_nodes = []

    def collect_leaf_nodes(node):
        if not node.children:
            visual_word_nodes.append(node)
        else:
            for child in node.children:
                collect_leaf_nodes(child)

    collect_leaf_nodes(root)

    # Assign visual words to database images
    image_visual_words = defaultdict(list)
    for obj_num in range(1, num_objects + 1):
        descriptors_path = os.path.join(database_dir, f'obj{obj_num}_descriptors.npy')
        descriptors = np.load(descriptors_path)
        visual_words = get_visual_words(descriptors, root)
        image_visual_words[obj_num].extend(visual_words)

    # Compute IDF values
    idf = compute_idf(num_objects, visual_word_nodes)

    # Build TF-IDF vectors for database images
    db_tfidf_vectors = build_tfidf_vectors(image_visual_words, idf, total_visual_words)

    # Process for each percentage
    for percent in percentages:
        print(f'\n--- Testing with {percent}% of Query Features ---\n')
        top1_correct = 0
        top5_correct = 0
        for obj_num in range(1, num_objects + 1):
            query_descriptors = query_features_dict[obj_num]
            # Determine the number of features to use
            num_features = int(len(query_descriptors) * percent / 100)
            if num_features == 0:
                num_features = 1  # Ensure at least one feature is used
            # Randomly sample the features
            np.random.seed(0)  # For reproducibility
            indices = np.random.choice(len(query_descriptors), num_features, replace=False)
            sampled_descriptors = query_descriptors[indices]
            # Get visual words
            query_visual_words = get_visual_words(sampled_descriptors, root)
            query_visual_words_dict = {obj_num: query_visual_words}
            query_tfidf_vectors = build_tfidf_vectors(query_visual_words_dict, idf, total_visual_words)
            query_vector = query_tfidf_vectors[obj_num]
            ranked_list = rank_images(query_vector, db_tfidf_vectors)
            top_results = [img_id for img_id, score in ranked_list[:5]]

            # Check if the correct object is in top-1 and top-5
            if top_results[0] == obj_num:
                top1_correct += 1
            if obj_num in top_results:
                top5_correct += 1

            print(f'Query Object {obj_num}: Top-5 Results: {top_results}')

        # Calculate recall rates
        top1_recall = top1_correct / num_objects
        top5_recall = top5_correct / num_objects

        print(f'\nResults using {percent}% of Query Features:')
        print(f'Top-1 Recall Rate: {top1_recall * 100:.2f}%')
        print(f'Top-5 Recall Rate: {top5_recall * 100:.2f}%')

if __name__ == '__main__':
    part4_a()
    # part4_b()