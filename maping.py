import numpy as np
import matplotlib.pyplot as plt

# Define map parameters
map_width = 10  # width of the environment in meters
map_height = 10  # height of the environment in meters
grid_resolution = 0.1  # size of each cell in meters

# Initialize the occupancy grid
grid_rows = int(map_height / grid_resolution)
grid_cols = int(map_width / grid_resolution)
occupancy_grid = np.full((grid_rows, grid_cols), 0.5)  # Initialize with unknown probabilities

# Simulated robot movement and sensor readings
robot_position = (5.0, 5.0)  # Starting position of the robot
ir_readings = [1.5, 1.0, 2.2, 1.8, 1.6, 1.9, 2.1, 2.5]  # Simulated IR sensor readings

# Update occupancy grid based on sensor readings
for angle, ir_reading in enumerate(ir_readings):
    # Calculate position in the direction of the sensor reading
    sensor_x = robot_position[0] + ir_reading * np.cos(np.radians(angle * 45))
    sensor_y = robot_position[1] + ir_reading * np.sin(np.radians(angle * 45))
    
    # Convert position to grid coordinates
    grid_row = int(sensor_y / grid_resolution)
    grid_col = int(sensor_x / grid_resolution)
    
    # Update occupancy grid
    if 0 <= grid_row < grid_rows and 0 <= grid_col < grid_cols:
        occupancy_grid[grid_row][grid_col] = 0.9 if ir_reading < 1.0 else 0.1

# Visualize the occupancy grid
plt.imshow(occupancy_grid, cmap='gray', origin='lower', extent=(0, map_width, 0, map_height))
plt.colorbar()
plt.scatter(robot_position[0], robot_position[1], color='red', label='Robot Position')
plt.xlabel('X (meters)')
plt.ylabel('Y (meters)')
plt.title('Occupancy Grid Map with IR Readings')
plt.legend()
plt.show()
