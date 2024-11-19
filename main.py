import pandas as pd
import random
import math
import matplotlib.pyplot as plt
from copy import deepcopy

# Step 1: Define the distance function
def euclidean_distance(point1, point2):
    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))

# Step 2: Initialize centroids randomly
def initialize_centroids(data, k):
    return random.sample(data, k)

# Step 3: Assign points to the nearest cluster
def assign_clusters(data, centroids):
    clusters = [[] for _ in centroids]
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        closest_cluster = distances.index(min(distances))
        clusters[closest_cluster].append(point)
    return clusters

# Step 4: Update centroids
def update_centroids(clusters):
    new_centroids = []
    for cluster in clusters:
        if len(cluster) > 0:
            new_centroid = [sum(dim) / len(cluster) for dim in zip(*cluster)]
        else:
            new_centroid = random.choice(cluster)
        new_centroids.append(new_centroid)
    return new_centroids

# Step 5: Calculate inertia
def calculate_inertia(clusters, centroids):
    inertia = 0
    for i, cluster in enumerate(clusters):
        for point in cluster:
            inertia += euclidean_distance(point, centroids[i]) ** 2
    return inertia

# Step 6: K-Means algorithm with Elbow Method visualization
def k_means_with_elbow(data, max_k=10, max_iterations=100, tolerance=1e-4):
    inertias_per_k = []

    plt.figure(figsize=(10, 6))

    for k in range(1, max_k + 1):
        centroids = initialize_centroids(data, k)
        prev_centroids = None
        iteration = 0

        for _ in range(max_iterations):
            iteration += 1
            clusters = assign_clusters(data, centroids)
            new_centroids = update_centroids(clusters)

            # Visualize each iteration
            plt.clf()
            colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'brown', 'pink', 'gray', 'olive']
            for i, cluster in enumerate(clusters):
                cluster_x = [point[0] for point in cluster]
                cluster_y = [point[1] for point in cluster]
                plt.scatter(cluster_x, cluster_y, color=colors[i % len(colors)], alpha=0.6, label=f"Cluster {i+1}")
            
            centroid_x = [centroid[0] for centroid in centroids]
            centroid_y = [centroid[1] for centroid in centroids]
            plt.scatter(centroid_x, centroid_y, color='black', marker='X', s=200, label='Centroids')

            plt.title(f"K-Means Clustering - k={k}, Iteration {iteration}")
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            plt.legend()
            plt.pause(0.5)  # Pause to display each iteration graph
            
            # Check for convergence
            if prev_centroids and all(euclidean_distance(c, nc) < tolerance for c, nc in zip(centroids, new_centroids)):
                break
            prev_centroids = deepcopy(centroids)
            centroids = deepcopy(new_centroids)
        
        # Calculate and store inertia for this k
        inertia = calculate_inertia(clusters, centroids)
        inertias_per_k.append(inertia)

    # Plot the Elbow Method curve
    plt.figure()
    plt.plot(range(1, max_k + 1), inertias_per_k, marker='o')
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.grid()
    plt.show()

    return inertias_per_k

# Step 7: Load and preprocess CSV data
def load_and_preprocess_csv(file_path):
    data = pd.read_csv(file_path)
    data['type_encoded'] = data['type'].astype('category').cat.codes
    features = data[['MA', 'bedrooms', 'type_encoded']].values.tolist()
    return features

# Step 8: Test with CSV Data
if __name__ == "__main__":
    file_path = 'ma_lga_12345.csv'
    data = load_and_preprocess_csv(file_path)
    max_k = 5
    inertias = k_means_with_elbow(data, max_k=max_k)

