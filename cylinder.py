import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

lidar_data = np.loadtxt("/home/atharva/Documents/MTP_2_001/tree_species_classification/4_LidarTreePoinCloudData/GUY01_000.txt")

def calculate_error_cylinder(axis, center, radius, points):
    """
    Calculate the total error based on the difference between the measured points and the cylinder surface.
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    if axis == 'X':
        error = np.sum((y - center[1])**2 + (z - center[2])**2 - radius**2)
    elif axis == 'Y':
        error = np.sum((x - center[0])**2 + (z - center[2])**2 - radius**2)
    elif axis == 'Z':
        error = np.sum((x - center[0])**2 + (y - center[1])**2 - radius**2)
    return error

def optimize_cylinder(points, axis, initial_center, initial_radius, alpha, Nloop, Nopt, Emin):
    """
    Optimize the cylinder fitting based on the finite random search algorithm.
    """
    current_center = np.array(initial_center)
    current_radius = initial_radius
    current_error = calculate_error_cylinder(axis, current_center, current_radius, points)
    
    for iteration in range(Nloop):
        for opt in range(Nopt):
            # Generate random adjustments
            adjustments = np.random.rand(3) * 2 - 1  # Random values in [-1, 1] for center
            radius_adjustment = np.random.rand() * 2 - 1  # Random value in [-1, 1] for radius
            
            # Scale adjustments
            adjustments *= alpha
            radius_adjustment *= alpha
            
            # Apply adjustments
            new_center = current_center + adjustments
            new_radius = current_radius + radius_adjustment
            
            # Calculate new error
            new_error = calculate_error_cylinder(axis, new_center, new_radius, points)
            
            # Update current values if error is reduced
            if new_error < current_error:
                current_center, current_radius, current_error = new_center, new_radius, new_error
                
                # Early stopping if error is below threshold
                if current_error < Emin:
                    return current_center, current_radius, current_error
        
        # Reduce alpha (constraint space) after each outer loop iteration
        alpha *= 0.95
    
    return current_center, current_radius, current_error

# Define functions for cylinder fitting
def construct_initial_constraint_space_cylinder(points, Rset):
    """
    Construct the initial constraint space based on the point cloud and preset parameter Rset.
    """
    # Calculate centroid of the point cloud
    centroid = np.mean(points, axis=0)
    
    # Determine ranges for X, Y, Z axes and radius
    X_range = [centroid[0] - 2 * Rset, centroid[0] + 2 * Rset]
    Y_range = [centroid[1] - 2 * Rset, centroid[1] + 2 * Rset]
    Z_range = [centroid[2] - 2 * Rset, centroid[2] + 2 * Rset]
    R_range = [0, 2 * Rset]
    
    # Construct initial constraint space
    initial_constraint_space = {
        'X': X_range,
        'Y': Y_range,
        'Z': Z_range,
        'R': R_range
    }
    
    return initial_constraint_space

def update_constraint_space_cylinder(axis, current_center, current_radius, current_constraint_space, alpha):
    """
    Update the constraint space based on the current center, radius, and scaling factor.
    """
    new_X_range = [current_center[0] - alpha * current_radius, current_center[0] + alpha * current_radius]
    new_Y_range = [current_center[1] - alpha * current_radius, current_center[1] + alpha * current_radius]
    new_Z_range = [current_center[2] - alpha * current_radius, current_center[2] + alpha * current_radius]
    new_R_range = [0, alpha * current_radius]
    
    updated_constraint_space = {
        'X': new_X_range,
        'Y': new_Y_range,
        'Z': new_Z_range,
        'R': new_R_range
    }
    
    return updated_constraint_space

def generate_sample_space(initial_constraint_space, Nloop):
    """
    Generate the sample space based on the initial constraint space.
    """
    X_min, X_max = initial_constraint_space['X']
    Y_min, Y_max = initial_constraint_space['Y']
    Z_min, Z_max = initial_constraint_space['Z']
    R_min, R_max = initial_constraint_space['R']
    
    # Generate random samples within the ranges
    X_samples = np.random.uniform(X_min, X_max, Nloop)
    Y_samples = np.random.uniform(Y_min, Y_max, Nloop)
    Z_samples = np.random.uniform(Z_min, Z_max, Nloop)
    R_samples = np.random.uniform(R_min, R_max, Nloop)
    
    sample_space = np.column_stack((X_samples, Y_samples, Z_samples, R_samples))
    
    return sample_space

# Define optimization parameters
alpha = 0.1  # Scaling factor for updating constraint space
Nloop = 100  # Number of samples in each iteration
Nopt = 30  # Number of optimization iterations
Emin = 1e-6  # Minimum error threshold

# Generate initial constraint space for cylinder fitting
initial_constraint_space_cylinder = construct_initial_constraint_space_cylinder(lidar_data, Rset=1.0)

# Initialize variables for optimization
current_constraint_space_cylinder = initial_constraint_space_cylinder
best_error_cylinder = float('inf')  # Initialize with infinity
best_center_cylinder = None
best_radius_cylinder = None

# Start optimization loop for cylinder fitting
for opt in range(Nopt):
    # Generate sample space
    sample_space = generate_sample_space(current_constraint_space_cylinder, Nloop)
    
    # Iterate through samples in sample space
    for sample in sample_space:
        # Extract center and radius from sample
        sample_center = sample[:3]
        sample_radius = sample[3]
        
        # Optimize cylinder fitting for the current sample
        optimized_center, optimized_radius, error = optimize_cylinder(lidar_data, 'X', sample_center, sample_radius, alpha, Nloop, 1, Emin)
        
        # Update best solution if error is improved
        if error < best_error_cylinder:
            best_error_cylinder = error
            best_center_cylinder = optimized_center
            best_radius_cylinder = optimized_radius
            
    # Update constraint space for next iteration
    current_constraint_space_cylinder = update_constraint_space_cylinder('X', best_center_cylinder, best_radius_cylinder, current_constraint_space_cylinder, alpha)

# Print optimized parameters and error for cylinder fitting
print(f"Optimized center: {best_center_cylinder}, Optimized radius: {best_radius_cylinder}, Final error: {best_error_cylinder}")

# Plotting the fitted cylinder
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Original points
ax.scatter(lidar_data[:, 0], lidar_data[:, 1], lidar_data[:, 2], color='b', label='Original Points')

# Plotting cylinder wireframe
u = np.linspace(0, 2 * np.pi, 100)
z = np.linspace(best_center_cylinder[2] - best_radius_cylinder, best_center_cylinder[2] + best_radius_cylinder, 100)
x = best_center_cylinder[0] + best_radius_cylinder * np.cos(u)
y = best_center_cylinder[1] + best_radius_cylinder * np.sin(u)
X, Z = np.meshgrid(x, z)
Y, _ = np.meshgrid(y, z)
ax.plot_surface(X, Y, Z, color='r', alpha=0.2)

# Set plot limits based on cylinder dimensions
max_range_cylinder = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
mid_x_cylinder = (X.max()+X.min()) * 0.5
mid_y_cylinder = (Y.max()+Y.min()) * 0.5
mid_z_cylinder = (Z.max()+Z.min()) * 0.5
ax.set_xlim(mid_x_cylinder - max_range_cylinder, mid_x_cylinder + max_range_cylinder)
ax.set_ylim(mid_y_cylinder - max_range_cylinder, mid_y_cylinder + max_range_cylinder)
ax.set_zlim(mid_z_cylinder - max_range_cylinder, mid_z_cylinder + max_range_cylinder)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Fitting Cylinder to Point Cloud Data')

plt.show()