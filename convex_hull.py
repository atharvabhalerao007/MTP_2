import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import alphashape
import pandas as pd
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
import itertools
import string

class PointCloudProcessor:
    def __init__(self, point_cloud_dir, output_dir):
        self.point_cloud_dir = point_cloud_dir
        self.output_dir = output_dir

    def load_point_cloud(self, file_path):
        try:
            return np.loadtxt(file_path)
        except Exception as e:
            logging.error(f"Error loading point cloud from {file_path}: {e}")
            return None

    def downsample_point_cloud(self, points, voxel_size=0.05):
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            down_pcd = pcd.voxel_down_sample(voxel_size)
            return np.asarray(down_pcd.points)
        except Exception as e:
            logging.error(f"Error downsampling point cloud: {e}")
            return None

    def filter_outliers(self, points, nb_neighbors=20, std_ratio=2.0):
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
            inlier_pcd = pcd.select_by_index(ind)
            return np.asarray(inlier_pcd.points)
        except Exception as e:
            logging.error(f"Error filtering outliers: {e}")
            return points

    def filter_canopy_points(self, points, height_threshold):
        try:
            return points[points[:, 2] >= height_threshold]
        except Exception as e:
            logging.error(f"Error filtering canopy points: {e}")
            return points

    def cluster_points(self, points, eps=0.5, min_samples=10):
        try:
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
            labels = clustering.labels_
            unique_labels = set(labels)
            clusters = []
            for label in unique_labels:
                if label == -1:
                    continue
                cluster_points = points[labels == label]
                clusters.append(cluster_points)
            return clusters
        except Exception as e:
            logging.error(f"Error clustering points: {e}")
            return []

    def validate_tree_clusters(self, clusters, min_height=1.5, max_height=50.0, min_density=0.0005):
        validated_clusters = []
        for cluster in clusters:
            if cluster.shape[0] < 3:
                continue
            height = np.max(cluster[:, 2]) - np.min(cluster[:, 2])
            if height < min_height or height > max_height:
                continue
            try:
                density = len(cluster) / ConvexHull(cluster[:, :2]).volume
                if density >= min_density:
                    validated_clusters.append(cluster)
                else:
                    logging.info(f"Cluster density too low: {density}")
            except Exception as e:
                logging.warning(f"Error computing convex hull for density calculation: {e}")
        return validated_clusters

    def add_noise_to_points(self, points, noise_level=1e-5):
        try:
            noise = np.random.normal(scale=noise_level, size=points.shape)
            return points + noise
        except Exception as e:
            logging.error(f"Error adding noise to points: {e}")
            return points

    def fit_cone(self, points):
        def cone_residuals(params):
            xc, yc, zc, h, r = params
            distances = np.sqrt((points[:, 0] - xc)**2 + (points[:, 1] - yc)**2)
            predicted_r = r * (zc - points[:, 2]) / h
            residuals = distances - predicted_r
            return np.sum(residuals**2)

        try:
            z_max = np.max(points[:, 2])
            z_min = np.min(points[:, 2])
            r_guess = np.mean(np.sqrt((points[:, 0] - np.mean(points[:, 0]))**2 + (points[:, 1] - np.mean(points[:, 1]))**2))
            initial_params = [np.mean(points[:, 0]), np.mean(points[:, 1]), z_max, z_max - z_min, r_guess]
            bounds = [(None, None), (None, None), (z_max, z_max), (0, z_max - z_min), (0, None)]
            result = minimize(cone_residuals, initial_params, bounds=bounds)
            return result.x
        except Exception as e:
            logging.error(f"Error fitting cone: {e}")
            return initial_params

    def compute_external_features(self, file_path):
        points = self.load_point_cloud(file_path)
        if points is None:
            return []

        downsampled_points = self.downsample_point_cloud(points)
        if downsampled_points is None:
            return []

        # Remove outliers from the point cloud
        filtered_points = self.filter_outliers(downsampled_points)

        # Cluster the points to identify individual trees
        clusters = self.cluster_points(filtered_points)
        validated_clusters = self.validate_tree_clusters(clusters)

        all_features = []
        tree_labels = self.generate_tree_labels()
        for canopy_points in validated_clusters:
            if canopy_points.shape[0] < 3:
                continue

            tree_label = next(tree_labels)

           
            height_threshold = np.percentile(canopy_points[:, 2], 70)
            canopy_points = self.filter_canopy_points(canopy_points, height_threshold)

            # Add noise to avoid precision issues
            canopy_points = self.add_noise_to_points(canopy_points)

            # Convex Hull
            try:
                hull = ConvexHull(canopy_points)
                hull_volume = hull.volume
                canopy_area = hull.area
            except Exception as e:
                logging.error(f"Error computing convex hull for tree {tree_label}: {e}")
                continue

            # Fit Cone
            cone_params = self.fit_cone(canopy_points)
            cone_height = cone_params[3]
            cone_radius = cone_params[4]
            cone_volume = (1/3) * np.pi * (cone_radius ** 2) * cone_height

            # Alpha Shape
            try:
                alpha_shape_geom = alphashape.alphashape(canopy_points, alpha=0.1)
                alpha_shape_volume = alpha_shape_geom.volume
            except Exception as e:
                logging.error(f"Error computing alpha shape for tree {tree_label}: {e}")
                alpha_shape_volume = 0

            # Additional Features
            canopy_density = canopy_points.shape[0] / canopy_area if canopy_area != 0 else 0

            # T_v: Volume of the convex hull divided by the number of points within the crown
            T_v = hull_volume / canopy_points.shape[0]

            # T_d: Difference between the convex hull and fitted cone volumes compared to the convex hull volume
            T_d = (hull_volume - cone_volume) / hull_volume

            # T_e: Root mean squared error from regression fitting of cone
            cone_fitted_points = np.column_stack((canopy_points[:, :2], cone_params[3] - (cone_radius * np.linalg.norm(canopy_points[:, :2] - cone_params[:2], axis=1) / cone_radius)))
            T_e = np.sqrt(np.mean((canopy_points[:, 2] - cone_fitted_points[:, 2])**2))

            # T_l: Average of distance d_n of each LiDAR point to the closest facet of convex hull
            hull_eq = hull.equations[:, :3]
            hull_d = hull.equations[:, 3]
            distances = np.abs(np.dot(canopy_points, hull_eq.T) + hull_d) / np.linalg.norm(hull_eq, axis=1)
            T_l = np.mean(distances)

            # T_σ: Standard deviation of orthogonal distances from each point to the convex hull
            T_σ = np.std(distances)

            # T_h: Crown height divided by Tree height
            tree_height = np.max(filtered_points[:, 2])
            T_h = (np.max(canopy_points[:, 2]) - np.min(canopy_points[:, 2])) / tree_height

            features = {
                'filename': os.path.basename(file_path),
                'tree_label': tree_label,
                'hull_volume': hull_volume,
                'cone_volume': cone_volume,
                'alpha_shape_volume': alpha_shape_volume,
                'canopy_area': canopy_area,
                'canopy_density': canopy_density,
                'T_v': T_v,
                'T_d': T_d,
                'T_e': T_e,
                'T_l': T_l,
                'T_σ': T_σ,
                'T_h': T_h
            }
            
            all_features.append(features)

        return all_features

   

    def save_features_to_csv(self, features, file_path):
        df = pd.DataFrame(features)
        if not os.path.isfile(file_path):
            df.to_csv(file_path, index=False)
        else:
            df.to_csv(file_path, header=False, index=False)
        logging.info(f"Features saved to {file_path}")

    def process_all_files(self, num_files=10):
        processed_files = 0
        for filename in os.listdir(self.point_cloud_dir):
            if filename.endswith(".txt"):
                if processed_files >= num_files:
                    break
                file_path = os.path.join(self.point_cloud_dir, filename)
                features = self.compute_external_features(file_path)
                if features is not None and len(features) > 0:
                    output_file = os.path.join(self.output_dir, 'external_features.csv')
                    self.save_features_to_csv(features, output_file)
                processed_files += 1

    def visualize_features(self, csv_file):
        try:
            df = pd.read_csv(csv_file)
            sns.pairplot(df)
            plt.savefig(os.path.join(self.output_dir, 'data_correlation_visualization.jpg'))
            plt.show()
        except Exception as e:
            logging.error(f"Error visualizing features: {e}")

def main(point_cloud_dir, output_dir, num_files=380):
    logging.basicConfig(level=logging.INFO)
    processor = PointCloudProcessor(point_cloud_dir, output_dir)
    processor.process_all_files(num_files)
    
    # Visualize the features
    output_file = os.path.join(output_dir, 'external_features.csv')
    processor.visualize_features(output_file)

if __name__ == "__main__":
    # Replace these paths with your actual directories
    point_cloud_dir = "/home/atharva/Documents/MTP_2_001/tree_species_classification/4_LidarTreePoinCloudData"
    output_dir = '/home/atharva/Documents/MTP_2_001/output001/789'
    main(point_cloud_dir, output_dir, num_files=29)