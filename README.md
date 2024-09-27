## code for EQ2425 KTH

## Requirements

- Python 3.x
- NumPy
- OpenCV (for SIFT feature extraction)
- SciPy (for hierarchical K-means)
- Any other libraries specified in `requirements.txt`

## Setup

1. **Place the Data:**

   First, put `Data2` file in the main directory, which contains the database and query images.

2. Then

   ```bash
   python pre_process.py
   python main.py

## Vocabulary Tree Construction
### (a)
- Centroid of the Cluster:
  - It allows us to compute the distance between the query feature and the cluster center.
- Child Nodes:
- Level of a nodes: to know the depth of search.
### (b)
- A dictionary with visual term (key points), to count (TF).
- The number of images in which the visual word appears (DF).
- Image ID

## Querying

[//]: # (### &#40;a&#41;)

[//]: # (Error Message:)

[//]: # (ValueError: n_samples=4 should be >= n_clusters=5.)

[//]: # (Solution:)

[//]: # (When building a vocabulary tree with a high branching factor &#40;b&#41; and depth &#40;depth&#41;, )

[//]: # (the number of data points at the deeper levels of the tree becomes very small. )

[//]: # (This is because the data is split into increasingly smaller subsets at each level of the tree. )

[//]: # (Eventually, you may reach a point where the number of data points is less than b.)

[//]: # ()
[//]: # (### &#40;b&#41;)

### (c)
For data-base with N key points, and query images with M key points.

- Traditional K-means Cluster
Without hierarchical clustering, each query feature would need to be compared to all cluster centroids or all features in the database.
This is computationally expensive for large N.
  -  Total number of Euclidean distance calculation is: O(N * M).
- Hierarchical K-means Cluster
Tree structure reduce the number of comparisons needed.
Let b be the branching factor (number of clusters), L be the depth of the tree. At each level, a query feature compares to b centroids.
  -  Total number of Euclidean distance calculation is: O(M * b * L)
