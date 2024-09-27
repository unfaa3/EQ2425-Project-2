import cv2
import numpy as np
import os

def extract_and_save_sift_features(image_paths, save_path, sift_detector):
    """
    Extracts SIFT features from a list of image paths and saves the combined descriptors.

    Parameters:
    - image_paths (list of str): List of paths to the images.
    - save_path (str): Path to save the combined descriptors.
    - sift_detector: Initialized SIFT detector.

    Returns:
    - total_features (int): Total number of features extracted from the images.
    """
    descriptors_list = []
    total_features = 0

    for img_path in image_paths:
        # Read the image in grayscale mode
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Check if the image was loaded properly
        if img is None:
            print(f'Warning: Image {img_path} could not be loaded.')
            continue

        # Detect SIFT features and compute descriptors
        keypoints, descriptors = sift_detector.detectAndCompute(img, None)

        # Check if descriptors were found
        if descriptors is not None:
            descriptors_list.append(descriptors)
            num_features = descriptors.shape[0]
            total_features += num_features
            print(f'{os.path.basename(img_path)}: {num_features} features extracted.')
        else:
            print(f'Warning: No features found in {os.path.basename(img_path)}.')

    if descriptors_list:
        # here vstack will stack along x-axis
        # which means x-axis will be (number of features * number of images)
        # y-axis will be 128 SIFT features
        combined_descriptors = np.vstack(descriptors_list)
        np.save(save_path, combined_descriptors)
        return total_features
    else:
        print(f'Warning: No descriptors were extracted from images in {save_path}.')
        return 0

def main():
    # Paths to the directories
    database_dir = 'Data2/server'
    query_dir = 'Data2/client'

    # Initialize SIFT detector
    sift = cv2.SIFT_create(contrastThreshold=0.17, edgeThreshold=10)

    # Variables to keep track of total features
    total_db_features = 0
    total_query_features = 0

    # Number of objects
    num_objects = 50

    print('--- Processing Database Images ---\n')
    # Process database images
    for obj_num in range(1, num_objects + 1):
        # Prepare list of image paths for the current object
        image_paths = [os.path.join(database_dir, f'obj{obj_num}_{img_num}.JPG') for img_num in range(1, 4)]
        # Path to save the combined descriptors
        save_path = os.path.join(database_dir, f'obj{obj_num}_descriptors.npy')
        # Extract and save features
        num_features = extract_and_save_sift_features(image_paths, save_path, sift)
        total_db_features += num_features
        # print(f'Object {obj_num}: {num_features} features extracted.\n')

    # Calculate and report the average number of features per object in the database
    average_db_features = total_db_features / num_objects
    print(f'Average number of features per database object: {average_db_features:.2f}\n')

    print('--- Processing Query Images ---\n')
    # Process query images
    for obj_num in range(1, num_objects + 1):
        # Path to the query image
        img_path = os.path.join(query_dir, f'obj{obj_num}_t1.JPG')
        # Path to save the descriptors
        save_path = os.path.join(query_dir, f'obj{obj_num}_descriptors.npy')
        # Extract and save features
        num_features = extract_and_save_sift_features([img_path], save_path, sift)
        total_query_features += num_features
        print(f'Query Object {obj_num}: {num_features} features extracted.\n')

    # Calculate and report the average number of features per query object
    average_query_features = total_query_features / num_objects
    print(f'Average number of features per database object: {average_db_features:.2f}\n')
    print(f'Average number of features per query object: {average_query_features:.2f}')

if __name__ == '__main__':
    main()