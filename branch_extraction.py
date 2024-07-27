import os
import sys
import numpy as np
import vtk
import open3d as o3d
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import logging
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from scipy.spatial import ConvexHull, cKDTree
from typing import List, Tuple, Dict, Callable
import argparse
import networkx as nx
import matplotlib.pyplot as plt

# Add the local mistree library to sys.path
mistree_path = '/home/atharva/Documents/MTP_2_001/tree_species_classification/mistree'
sys.path.insert(0, mistree_path)

# Ensure the compiled modules are in the path
mst_path = '/home/atharva/Documents/MTP_2_001/tree_species_classification/mistree/mistree/mst'
sys.path.insert(0, mst_path)

# Import compiled and necessary modules
from pc_skeletor.laplacian import LBC

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_point_cloud(file_path: str) -> np.ndarray:
    try:
        data = np.loadtxt(file_path)
        return data[:, :3]
    except IOError as e:
        logging.error(f"Error loading point cloud from {file_path}: {e}")
        return np.array([])

def create_vtk_points(points: np.ndarray) -> vtk.vtkPoints:
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(points))
    return vtk_points

def preprocess_point_cloud(points: np.ndarray, voxel_size: float) -> np.ndarray:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    downsampled_points = np.asarray(downsampled_pcd.points)
    return downsampled_points

def adaptive_graph_threshold(points, base_threshold=0.1):
    density = len(points) / np.ptp(points, axis=0).prod()
    return base_threshold * np.cbrt(density)

def detect_trunk(points, z_threshold=0.2, degree=2):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    trunk_points = points[inliers]
    trunk_points = trunk_points[trunk_points[:, 2] < z_threshold]
    model = make_pipeline(PolynomialFeatures(degree), RANSACRegressor())
    model.fit(trunk_points[:, :2], trunk_points[:, 2])
    return trunk_points, model

def construct_graph(points):
    tree = o3d.geometry.KDTreeFlann(points)
    G = nx.Graph()
    for i in range(len(points)):
        G.add_node(i, pos=points[i])
        _, idx, _ = tree.search_radius_vector_3d(points[i], adaptive_graph_threshold(points))
        for j in idx:
            if i != j:
                G.add_edge(i, j, weight=np.linalg.norm(points[i] - points[j]))
    return G

def hierarchical_clustering(points, n_clusters=5):
    cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
    labels = cluster.fit_predict(points)
    return labels

def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

def region_growing(points: np.ndarray, tips: np.ndarray, distance_threshold: float = 0.05, min_branch_size: int = 0, max_angle: float = 45) -> List[Tuple[np.ndarray, str]]:
    try:
        tree = cKDTree(points)
        labels = np.full(len(points), -1, dtype=int)
        current_label = 0
        branches = []

        for tip in tips:
            if labels[tree.query(tip)[1]] == -1:
                queue = [tree.query(tip)[1]]
                branch_points = [points[queue[0]]]
                labels[queue[0]] = current_label
                branch_direction = None
                is_trunk = current_label == 0

                while queue:
                    point_idx = queue.pop(0)
                    neighbors = tree.query_ball_point(points[point_idx], distance_threshold)

                    for neighbor_idx in neighbors:
                        if labels[neighbor_idx] == -1:
                            new_direction = points[neighbor_idx] - points[point_idx]

                            if branch_direction is None:
                                branch_direction = new_direction
                            elif angle_between(branch_direction, new_direction) <= max_angle:
                                labels[neighbor_idx] = current_label
                                queue.append(neighbor_idx)
                                branch_points.append(points[neighbor_idx])
                                branch_direction = new_direction  # Update branch direction

                if len(branch_points) >= min_branch_size:
                    branch_type = 'trunk' if is_trunk else 'branch'
                    branches.append((np.array(branch_points), branch_type))
                    current_label += 1

        return branches
    except Exception as e:
        logging.error(f"Error in region growing: {e}")
        return []

def extract_branch_tips(skeleton_graph: nx.Graph, pos_attribute: str = 'pos', degree_range: Tuple[int, int] = (1, 3), custom_criteria: Callable = None, min_branch_length: float = 0.0, angle_threshold: float = 45.0) -> np.ndarray:
    """
    Extract branch tips from a skeleton graph with flexible criteria.
    
    Args:
    skeleton_graph (nx.Graph): The input skeleton graph.
    pos_attribute (str): The attribute name for node positions.
    degree_range (Tuple[int, int]): The range of degrees to consider as potential tips.
    custom_criteria (Callable): Optional custom function to determine if a node is a tip.
    min_branch_length (float): Minimum length of a branch to be considered a tip.
    angle_threshold (float): Minimum angle (in degrees) between branches to be considered a split.
    
    Returns:
    np.ndarray: Array of tip positions.
    """
    try:
        tips = []
        for node, data in skeleton_graph.nodes(data=True):
            if pos_attribute not in data:
                raise ValueError(f"Node {node} does not have '{pos_attribute}' attribute.")
            
            pos = np.array(data[pos_attribute])
            degree = skeleton_graph.degree(node)
            
            # Basic degree check
            if degree_range[0] <= degree <= degree_range[1]:
                # Apply custom criteria if provided
                if custom_criteria and not custom_criteria(skeleton_graph, node):
                    continue
                
                # Check branch length
                if degree == 1:
                    neighbor = list(skeleton_graph.neighbors(node))[0]
                    branch_length = np.linalg.norm(pos - np.array(skeleton_graph.nodes[neighbor][pos_attribute]))
                    if branch_length < min_branch_length:
                        continue
                
                # Check angle for branching points
                if degree > 2:
                    angles = calculate_branch_angles(skeleton_graph, node, pos_attribute)
                    if max(angles) < angle_threshold:
                        continue
                
                tips.append(pos)
            
        return np.array(tips)
    
    except Exception as e:
        logging.error(f"Error in extract_branch_tips: {e}")
        return np.empty((0, 3))

def calculate_branch_angles(graph: nx.Graph, node: int, pos_attribute: str) -> List[float]: 
    """Calculate angles between branches at a given node."""
    neighbors = list(graph.neighbors(node))
    if len(neighbors) < 2:
        return []
    
    node_pos = np.array(graph.nodes[node][pos_attribute])
    vectors = [np.array(graph.nodes[n][pos_attribute]) - node_pos for n in neighbors]
    
    angles = []
    for i in range(len(vectors)):
        for j in range(i+1, len(vectors)):
            angle = np.degrees(np.arccos(np.dot(vectors[i], vectors[j]) / 
                               (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]))))
            angles.append(angle)
    
    return angles

def calculate_branch_features(branches: List[Tuple[np.ndarray, str]], trunk_vector: np.ndarray, stem_vector: np.ndarray, max_length: float = 5, min_density: float = 0.01) -> Tuple[Dict[str, List[float]], List[np.ndarray]]:
    features = {
        "branch_id": [], "length": [], "slope": [], "compactness": [], "width": [],
        "symmetry": [], "density": [], "type": []
    }

    for branch_id, (branch_points, branch_type) in enumerate(branches):
        if len(branch_points) < 3:
            continue

        pca = PCA(n_components=3)
        pca.fit(branch_points)
        eigenvalues = pca.explained_variance_
        eigenvectors = pca.components_

        projected_points = pca.transform(branch_points)
        branch_length = projected_points[:, 0].max() - projected_points[:, 0].min()
        
        if branch_length > max_length:
            continue

        branch_width = projected_points[:, 1].max() - projected_points[:, 1].min()

        if len(branch_points) > 3 and not is_coplanar(branch_points):
            try:
                hull = ConvexHull(branch_points)
                branch_compactness = hull.volume / len(branch_points)
            except Exception:
                branch_compactness = 0
        else:
            branch_compactness = 0

        centroid = np.mean(branch_points, axis=0)
        distances = np.linalg.norm(branch_points - centroid, axis=1)
        radial_symmetry = np.std(distances) / np.mean(distances)
        axial_symmetry = np.std(distances) / (np.mean(distances) + np.std(distances))
        branch_symmetry = (radial_symmetry + axial_symmetry) / 2

        if len(branch_points) > 1:
            scaled_points = StandardScaler().fit_transform(branch_points)
            db = DBSCAN(eps=0.05, min_samples=3).fit(scaled_points)
            labels = db.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            branch_density = n_clusters_ / len(branch_points)
        else:
            branch_density = 0

        if branch_density < min_density:
            continue

        branch_vector = eigenvectors[0]
        branch_angle = np.arccos(np.clip(np.dot(branch_vector, trunk_vector) / (np.linalg.norm(branch_vector) * np.linalg.norm(trunk_vector)), -1.0, 1.0))
        slope = np.degrees(branch_angle)

        features["branch_id"].append(branch_id)
        features["length"].append(branch_length)
        features["slope"].append(slope)
        features["compactness"].append(branch_compactness)
        features["width"].append(branch_width)
        features["symmetry"].append(branch_symmetry)
        features["density"].append(branch_density)
        features["type"].append(branch_type)

    return features, [branch[0] for branch in branches]

def is_coplanar(points: np.ndarray) -> bool:
    if len(points) < 3:
        return True
    p0 = points[0]
    normal = np.cross(points[1] - p0, points[2] - p0)
    return all(np.isclose(np.dot(point - p0, normal), 0) for point in points[3:])

def detect_outliers(data: np.ndarray, threshold: float = 1.5) -> np.ndarray:
    if data.size == 0:
        return np.array([], dtype=int)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    return np.where((data < lower_bound) | (data > upper_bound))[0]

def create_vtk_lines(points: np.ndarray, color: Tuple[float, float, float], branch_id: int) -> vtk.vtkActor:
    lines = vtk.vtkCellArray()
    for i in range(len(points) - 1):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, i)
        line.GetPointIds().SetId(1, i + 1)
        lines.InsertNextCell(line)
    
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(create_vtk_points(points))
    polydata.SetLines(lines)
    
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colors.SetName("Colors")
    for _ in range(len(points) - 1):
        colors.InsertNextTuple3(color[0] * 255, color[1] * 255, color[2] * 255)
    polydata.GetCellData().SetScalars(colors)
    
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetLineWidth(2)
    actor.GetProperty().SetPointSize(4)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetOpacity(0.6)
    actor.SetPickable(True)
    actor.GetProperty().SetLineStipplePattern(branch_id)
    return actor

class BranchInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, parent=None):
        self.AddObserver("LeftButtonPressEvent", self.left_button_press_event)
        self.branches = []

    def left_button_press_event(self, obj, event):
        click_pos = self.GetInteractor().GetEventPosition()
        picker = vtk.vtkPropPicker()
        picker.Pick(click_pos[0], click_pos[1], 0, self.GetDefaultRenderer())
        actor = picker.GetActor()
        if actor:
            branch_id = actor.GetProperty().GetLineStipplePattern()
            print(f"Selected Branch ID: {branch_id}")
        self.OnLeftButtonDown()

def visualize_skeleton_and_branches_3d(skeleton_graph: nx.Graph, branches: List[Tuple[np.ndarray, str]], title: str):
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    style = BranchInteractorStyle()
    style.SetDefaultRenderer(renderer)
    interactor.SetInteractorStyle(style)

    if skeleton_graph is not None:
        skeleton_nodes = np.array([skeleton_graph.nodes[node]['pos'] for node in skeleton_graph.nodes])
        node_idx_map = {node: idx for idx, node in enumerate(skeleton_graph.nodes)}
        skeleton_edges = np.array([[node_idx_map[u], node_idx_map[v]] for u, v in skeleton_graph.edges])
        
        lines = vtk.vtkCellArray()
        for edge in skeleton_edges:
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, edge[0])
            line.GetPointIds().SetId(1, edge[1])
            lines.InsertNextCell(line)
        
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(create_vtk_points(skeleton_nodes))
        polydata.SetLines(lines)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0, 0, 0)  # Black color for skeleton
        actor.GetProperty().SetLineWidth(1)
        renderer.AddActor(actor)

    for i, (branch, branch_type) in enumerate(branches):
        color = (0, 1, 0) if branch_type == 'trunk' else (1, 0, 0)  # Green for trunk, Red for branches
        branch_actor = create_vtk_lines(branch, color, i)
        style.branches.append(branch_actor)
        renderer.AddActor(branch_actor)

    renderer.SetBackground(1, 1, 1)
    render_window.SetSize(800, 600)
    render_window.SetWindowName(title)
    render_window.Render()
    interactor.Start()

def dfs_traversal(graph: nx.Graph, start_node: int, visited: set) -> List[int]:
    branch = []
    stack = [start_node]

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            branch.append(node)
            neighbors = list(graph.neighbors(node))
            stack.extend(neighbors)

    return branch

def remove_small_components(graph: nx.Graph, size_threshold: int) -> nx.Graph:
    connected_components = list(nx.connected_components(graph))
    large_components = [comp for comp in connected_components if len(comp) >= size_threshold]
    new_graph = graph.subgraph([node for comp in large_components for node in comp]).copy()
    return new_graph

def extract_branches_using_graph(graph: nx.Graph) -> List[Tuple[np.ndarray, str]]:
    branches = []
    components = list(nx.connected_components(graph))
    for component in components:
        subgraph = graph.subgraph(component)
        root = min(subgraph, key=lambda n: subgraph.nodes[n]['pos'][2])
        visited = set()
        trunk_branch = dfs_traversal(subgraph, root, visited)
        branches.append((np.array([subgraph.nodes[node]['pos'] for node in trunk_branch]), 'trunk'))
        
        for node in trunk_branch:
            neighbors = list(subgraph.neighbors(node))
            for neighbor in neighbors:
                if neighbor not in visited:
                    branch = dfs_traversal(subgraph, neighbor, visited)
                    if branch:
                        branches.append((np.array([subgraph.nodes[n]['pos'] for n in branch]), 'branch'))
    
    branches = [(points, branch_type) for points, branch_type in branches if branch_type == 'branch']
    return branches

# New function to extract individual branches from the LBC skeleton
def extract_individual_branches(lbc, final_skeleton_graph):
    branches = []
    branch_ids = {}
    branch_count = 0

    # Identify branch endpoints and intersections
    for node, degree in final_skeleton_graph.degree():
        if degree == 1 or degree != 2:  # Endpoints and intersections
            branches.append((node, branch_count))
            branch_ids[node] = branch_count
            branch_count += 1

    # Perform BFS from each branch endpoint
    for start_node, branch_id in branches:
        queue = [(start_node, 0)]
        visited = {start_node}
        while queue:
            current_node, path_length = queue.pop(0)
            for neighbor in final_skeleton_graph.neighbors(current_node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path_length + 1))
                    branch_ids[neighbor] = branch_id

    # Group nodes by branch IDs
    branch_segments = {}
    for node, branch_id in branch_ids.items():
        if branch_id not in branch_segments:
            branch_segments[branch_id] = []
        branch_segments[branch_id].append(final_skeleton_graph.nodes[node]['pos'])

    # Fit cylinders or curves to the branch segments
    fitted_branches = []
    for branch_id, segment in branch_segments.items():
        segment = np.array(segment)
        if len(segment) > 1:
            # Fit a cylinder or curve (here simplified as PCA for fitting line)
            pca = PCA(n_components=1)
            pca.fit(segment)
            direction = pca.components_[0]
            fitted_branches.append((segment, direction))

    return fitted_branches

def process_file(file_path: str, output_dir: str):
    point_cloud_data = load_point_cloud(file_path)
    if point_cloud_data.size == 0:
        return

    downsampled_points = preprocess_point_cloud(point_cloud_data, voxel_size=0.05)
    
    logging.info(f"Type of downsampled_points: {type(downsampled_points)}")
    logging.info(f"Shape of downsampled_points: {downsampled_points.shape}")
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(downsampled_points)
    
    lbc = LBC(point_cloud=pcd, down_sample=0.008)
    lbc.extract_skeleton()
    lbc.extract_topology()

    filename = os.path.splitext(os.path.basename(file_path))[0]
    features_csv_path_region_growing = os.path.join(output_dir, f"{filename}_branch_features_region_growing.csv")
    features_csv_path_graph = os.path.join(output_dir, f"{filename}_branch_features_graph.csv")
    output_animation_path = os.path.join(output_dir, f"{filename}_animation")

    os.makedirs(output_animation_path, exist_ok=True)

    lbc.animate(init_rot=np.asarray([[1, 0, 0], [0, 0, 1], [0, 1, 0]]), steps=300, output=output_animation_path)
    logging.info(f"Processed {file_path}")

    if not hasattr(lbc, 'contracted_skeleton_graph') or lbc.contracted_skeleton_graph is None:
        logging.error("Contracted skeleton graph not found. Using regular skeleton graph.")
        final_skeleton_graph = lbc.skeleton_graph
    else:
        logging.info("Contracted skeleton graph found.")
        final_skeleton_graph = lbc.contracted_skeleton_graph

    logging.info(f"Regular skeleton graph - Nodes: {len(lbc.skeleton_graph.nodes)}, Edges: {len(lbc.skeleton_graph.edges)}")
    logging.info(f"Contracted skeleton graph - Nodes: {len(final_skeleton_graph.nodes)}, Edges: {len(final_skeleton_graph.edges)}")

    final_skeleton_graph = remove_small_components(final_skeleton_graph, size_threshold=50)

    contracted_skeleton_points = np.array([node['pos'] for node in final_skeleton_graph.nodes.values()])

    tips = extract_branch_tips(final_skeleton_graph)
    branches_region_growing = region_growing(contracted_skeleton_points, tips)
    branch_types_region_growing = [branch_type for _, branch_type in branches_region_growing]

    branches_graph = extract_branches_using_graph(final_skeleton_graph)
    branch_types_graph = [branch_type for _, branch_type in branches_graph]

    trunk_vector = np.array([0, 0, 1])
    stem_vector = np.array([1, 0, 0])

    features_region_growing, all_branches_region_growing = calculate_branch_features(branches_region_growing, trunk_vector, stem_vector)
    features_df_region_growing = pd.DataFrame(features_region_growing)

    features_graph, all_branches_graph = calculate_branch_features(branches_graph, trunk_vector, stem_vector)
    features_df_graph = pd.DataFrame(features_graph)

    if not features_df_region_growing.empty:
        features_df_region_growing.to_csv(features_csv_path_region_growing, index=False)
        logging.info(f"Features (Region Growing) saved to {features_csv_path_region_growing}")
    else:
        logging.info("No features (Region Growing) to save.")

    if not features_df_graph.empty:
        features_df_graph.to_csv(features_csv_path_graph, index=False)
        logging.info(f"Features (Graph) saved to {features_csv_path_graph}")
    else:
        logging.info("No features (Graph) to save.")

    # Extract individual branches from the LBC skeleton
    fitted_branches = extract_individual_branches(lbc, final_skeleton_graph)
    
    # Visualize all branches for both methods
    visualize_skeleton_and_branches_3d(final_skeleton_graph, branches_region_growing, title="Region Growing Branch Extraction")
    visualize_skeleton_and_branches_3d(final_skeleton_graph, branches_graph, title="Graph-Based Branch Extraction")
    # Additional visualization for fitted branches
    visualize_skeleton_and_branches_3d(final_skeleton_graph, fitted_branches, title="Fitted Branches from LBC Skeleton")

def main(num_files_to_process: int, input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".txt")]
    for file_path in files[:num_files_to_process]:
        process_file(file_path, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process point cloud data of trees.")
    parser.add_argument("--num_files", type=int, default=1, help="Number of files to process")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing point cloud files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for processed files")
    args = parser.parse_args()

    main(args.num_files, args.input_dir, args.output_dir)