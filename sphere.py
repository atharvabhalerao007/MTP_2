import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

lidar_data = np.loadtxt("/home/atharva/Documents/project/output0039.txt")

def calculate_error(center, radius, points):
    """
    Calculate the total error based on the difference between the measured points and the sphere surface.
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    error = np.sum((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 - radius**2)
    return error

def optimize_sphere(points, initial_center, initial_radius, alpha, Nloop, Nopt, Emin):
    """
    Optimize the sphere fitting based on the finite random search algorithm.
    """
    current_center = np.array(initial_center)
    current_radius = initial_radius
    current_error = calculate_error(current_center, current_radius, points)
    
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
            new_error = calculate_error(new_center, new_radius, points)
            
            # Update current values if error is reduced
            if new_error < current_error:
                current_center, current_radius, current_error = new_center, new_radius, new_error
                
                # Early stopping if error is below threshold
                if current_error < Emin:
                    return current_center, current_radius, current_error
        
        # Reduce alpha (constraint space) after each outer loop iteration
        alpha *= 0.95
    
    return current_center, current_radius, current_error

def construct_initial_constraint_space(points, Rset):
    """
    Construct the initial constraint space based on the point cloud and preset parameter Rset.
    """
    # Calculate centroid of the point cloud
    centroid = np.mean(points, axis=0)
    
    # Determine ranges for X, Y, Z axes and radius
    X_range = [np.min(points[:, 0]), np.max(points[:, 0])]
    Y_range = [np.min(points[:, 1]), np.max(points[:, 1])]
    Z_range = [np.min(points[:, 2]), np.max(points[:, 2])]
    
    # Limit radius based on maximum range of x, y, z
    R_range = [0, min(np.max(points[:, 0]) - np.min(points[:, 0]),
                      np.max(points[:, 1]) - np.min(points[:, 1]),
                      np.max(points[:, 2]) - np.min(points[:, 2]), 2 * Rset)]
    
    # Construct initial constraint space
    initial_constraint_space = {
        'X': X_range,
        'Y': Y_range,
        'Z': Z_range,
        'R': R_range
    }
    
    return initial_constraint_space

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

def update_constraint_space(current_center, current_radius, current_constraint_space, alpha):
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



# Define optimization parameters
alpha = 0.1  # Scaling factor for updating constraint space
Nloop = 100  # Number of samples in each iteration
Nopt = 20   # Number of optimization iterations
Emin = 1e-6  # Minimum error threshold

# Preset parameter for initial constraint space
Rset = 2.0  

# Example usage
points = lidar_data  # Use actual lidar data

# Calculate the mean of the data points
initial_radius = np.mean(np.linalg.norm(points - np.mean(points, axis=0), axis=1))

# Generate initial constraint space with tighter bounds
initial_constraint_space = construct_initial_constraint_space(points, initial_radius)

# Initialize variables for optimization
current_constraint_space = initial_constraint_space
best_error = float('inf')  # Initialize with infinity
best_center = None
best_radius = None

# Start optimization loop
for opt in range(Nopt):
    # Generate sample space
    sample_space = generate_sample_space(current_constraint_space, Nloop)
    
    # Iterate through samples in sample space
    for sample in sample_space:
        # Extract center and radius from sample
        sample_center = sample[:3]
        sample_radius = sample[3]
        
        # Optimize sphere fitting for the current sample
        optimized_center, optimized_radius, error = optimize_sphere(points, sample_center, sample_radius, alpha, Nloop, 1, Emin)
        
        # Update best solution if error is improved
        if error < best_error:
            best_error = error
            best_center = optimized_center
            best_radius = optimized_radius
            
# Limit the maximum radius of the fitted sphere based on the maximum range of x, y, z
max_radius = min(np.max(points[:, 0]) - np.min(points[:, 0]),
                 np.max(points[:, 1]) - np.min(points[:, 1]),
                 np.max(points[:, 2]) - np.min(points[:, 2]), best_radius)
best_radius = max_radius

# Calculate the percentage of points fitted by the sphere
fitted_points = np.sum(np.linalg.norm(points - best_center, axis=1) <= best_radius)
percentage_fitted = (fitted_points / len(points)) * 100

# Print optimized parameters, error, and percentage of points fitted by the sphere
print(f"Optimized center: {best_center}, Optimized radius: {best_radius}, Final error: {best_error}")
print(f"Percentage of points fitted by the sphere: {percentage_fitted:.2f}%")

# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Original points
ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='b', label='Original Points')

# Plotting sphere wireframe
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = best_radius * np.outer(np.cos(u), np.sin(v)) + best_center[0]
y = best_radius * np.outer(np.sin(u), np.sin(v)) + best_center[1]
z = best_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + best_center[2]
ax.plot_wireframe(x, y, z, color='r')

# Set plot limits based on sphere dimensions
max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
mid_x = (x.max()+x.min()) * 0.5
mid_y = (y.max()+y.min()) * 0.5
mid_z = (z.max()+z.min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Fitting Sphere to Point Cloud Data')

plt.legend()
plt.show()



