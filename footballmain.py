import pandas as pd  # For data manipulation and reading CSV files
import random  # For random initialization of centroids
import math  # For mathematical calculations like square root
import matplotlib.pyplot as plt  # For plotting the graphs
from copy import deepcopy  # For creating deep copies of objects

#step 1: Define the distance function
def euclidean_distance(point1, point2):
    """
    Calculates the Euclidean distance between two points
    """
    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))

#step 2: Initialize centroids randomly
def initialize_centroids(data, k):
    """
    Randomly selects `k` data points from the dataset as initial centroids
    """
    return random.sample(data, k)

#step 3: Assign points to the nearest cluster
def assign_clusters(data, centroids):
    """
    Assigns each data point to the nearest centroid based on the Euclidean distance
    """
    clusters = [[] for _ in centroids]  # Create empty lists for each cluster
    for point in data:
        #calculate the distance of the point to each centroid
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        #find the index of the closest centroid
        closest_cluster = distances.index(min(distances))
        #add the point to the corresponding cluster
        clusters[closest_cluster].append(point)
    return clusters

#step 4: Update centroids
def update_centroids(clusters):
    """
    Updates centroids by calculating the mean of all points in each cluster.
    """
    new_centroids = []
    for cluster in clusters:
        if len(cluster) > 0:
            #compute the mean for each dimension of the cluster
            new_centroid = [sum(dim) / len(cluster) for dim in zip(*cluster)]
        else:
            #handle empty clusters by choosing a random point
            new_centroid = random.choice(cluster)
        new_centroids.append(new_centroid)
    return new_centroids

#step 5: Calculate inertia
def calculate_inertia(clusters, centroids):
    """
    Calculates the inertia, which is the sum of squared distances of points to their centroids.
    """
    inertia = 0
    for i, cluster in enumerate(clusters):
        for point in cluster:
            #add the squared distance of each point to its centroid
            inertia += euclidean_distance(point, centroids[i]) ** 2
    return inertia

#step 6: K-Means algorithm with Elbow Method visualization and Name Display Option
def k_means_with_dynamic_labels(data, labels, x_label, y_label, display_names, max_k, max_iterations=100, tolerance=1e-4):
    """
    Performs K-Means clustering and visualizes the clusters for each k with the Elbow Method.
    Allows toggling of data point labels.
    """
    inertias_per_k = []  #to store inertia for each value of k

    plt.figure(figsize=(10, 6))  #create a new figure for the clustering visualization

    for k in range(1, max_k + 1):
        #randomly initialize centroids for the given value of k
        centroids = initialize_centroids(data, k)
        prev_centroids = None  #to track centroids for convergence
        iteration = 0

        for _ in range(max_iterations):
            iteration += 1
            #assign points to the nearest centroid
            clusters = assign_clusters(data, centroids)
            #update centroids based on the new clusters
            new_centroids = update_centroids(clusters)

            #visualize the current iteration
            plt.clf()  # Clear the figure
            colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'brown', 'pink', 'gray', 'olive']
            for i, cluster in enumerate(clusters):
                #extract x and y coordinates for the cluster
                cluster_x = [point[0] for point in cluster]
                cluster_y = [point[1] for point in cluster]
                #plot the cluster points
                plt.scatter(cluster_x, cluster_y, color=colors[i % len(colors)], alpha=0.6, label=f"Cluster {i+1}")
                
                #optionally display names on the points
                if display_names:
                    for point, label in zip(data, labels):
                        if point in cluster:
                            plt.text(point[0], point[1], label, fontsize=8, color='black')

            #plot the centroids
            centroid_x = [centroid[0] for centroid in centroids]
            centroid_y = [centroid[1] for centroid in centroids]
            plt.scatter(centroid_x, centroid_y, color='black', marker='X', s=200, label='Centroids')

            #add plot titles and labels
            plt.title(f"K-Means Clustering ({x_label} vs {y_label}) - k={k}, Iteration {iteration}")
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.legend()
            plt.pause(0.5)  #pause to allow the plot to render
            
            #check for convergence
            if prev_centroids and all(euclidean_distance(c, nc) < tolerance for c, nc in zip(centroids, new_centroids)):
                break
            prev_centroids = deepcopy(centroids)  #deep copy to track changes
            centroids = deepcopy(new_centroids)

        #calculate inertia for this value of k
        inertia = calculate_inertia(clusters, centroids)
        inertias_per_k.append(inertia)

    #plot the Elbow Method curve
    plt.figure()
    plt.plot(range(1, max_k + 1), inertias_per_k, marker='o')
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.grid()
    plt.show()

    return inertias_per_k

#step 7: Load Mall Customers Data
def load_mall_customer_data(file_path):
    """
    Loads the Mall Customers dataset and extracts features for clustering.
    """
    df = pd.read_csv(file_path)
    #extract features 'Annual Income (k$)' and 'Spending Score (1-100)'
    features = df[['Annual Income (k$)', 'Spending Score (1-100)']].dropna().values.tolist()
    #use CustomerID as labels for identifying points
    labels = df['CustomerID'].astype(str).tolist()
    return features, labels

#step 8: Load Football Data and Prompt for Features
def load_football_data(file_path):
    """
    Loads the Football dataset and prompts the user for feature selection.
    """
    df = pd.read_csv(file_path)
    print("Available columns in the dataset:", list(df.columns))  #show available columns
    feature_x = input("Enter the first feature to compare (default: 'targets'): ") or 'targets'
    feature_y = input("Enter the second feature to compare (default: 'receiving_yards'): ") or 'receiving_yards'

    #filter data for WRs in 2023 and extract selected features
    filtered_data = df[(df['season'] == 2023) & (df['position'] == 'WR')]
    features = filtered_data[[feature_x, feature_y]].dropna().values.tolist()
    labels = filtered_data['player_name'].tolist()  #use player names for labels
    return features, labels, feature_x, feature_y

#step 9: Main Program
if __name__ == "__main__":
    """
    Main program logic: Allows the user to choose which dataset to analyze.
    """
    print("Choose which dataset to analyze:")
    print("1. Mall Customers")
    print("2. Football Data")
    choice = input("Enter your choice (1 or 2): ")

    if choice == "1":
        print("Running K-Means on Mall Customers data...")
        file_path = input("Enter the path to the Mall Customers CSV file or press enter for default data: ") or 'Mall_Customers.csv'
        data, labels = load_mall_customer_data(file_path)
        display_names = input("Do you want to display names on the plot? (yes/no, default: yes): ").lower() in ['yes', 'y', '']
        max_k = input("What is the maximum amount of centroids you would like?") or 10
        k_means_with_dynamic_labels(data, labels, "Annual Income (k$)", "Spending Score", display_names,int(max_k))
    elif choice == "2":
        print("Running K-Means on Football data...")
        file_path = input("Enter the path to the football data CSV file or press enter for default data: ") or 'yearly_player_data.csv'
        data, labels, feature_x, feature_y = load_football_data(file_path)
        display_names = input("Do you want to display names on the plot? (yes/no, default: yes): ").lower() in ['yes', 'y', '']
        max_k = input("What is the maximum amount of centroids you would like?") or 10
        k_means_with_dynamic_labels(data, labels, feature_x, feature_y, display_names,int(max_k))
    else:
        print("Invalid choice. Exiting program.")

