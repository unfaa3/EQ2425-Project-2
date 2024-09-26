## code for EQ2425 KTH

## Image Feature Extraction
finished

## Vocabulary Tree Construction
### (a)
- Centroid of the Cluster:
  - It allows us to compute the distance between the query feature and the cluster center.
- Child Nodes:
- Level of a nodes: to know the depth of search.
### (b)
- A dictionary with visual term, to count (TF).
- The number of images in which the visual word appears (DF).

## Querying


### Error Message:
ValueError: n_samples=4 should be >= n_clusters=5.

When building a vocabulary tree with a high branching factor (b) and depth (depth), 
the number of data points at the deeper levels of the tree becomes very small. 
This is because the data is split into increasingly smaller subsets at each level of the tree. 
Eventually, you may reach a point where the number of data points is less than b.
