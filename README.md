#ultimate goal is to detect heat at night
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage import median_filter
import cv2
# Grid size
grid_size = 100

# Background temperature range
background_temp_min = 20
background_temp_max = 40

# Target temperature range
target_temp_mean = 35
target_temp_std = 2

# Number of targets
num_targets = 5

# Generate background temperature grid
background_temp = np.random.uniform(background_temp_min, background_temp_max, (grid_size, grid_size))
background_temp_smooth = gaussian_filter(background_temp, sigma=1.0)
background_temp_smooth = median_filter(background_temp, size=3)
background_temp_float = background_temp.astype(np.float32)
background_temp_smooth = cv2.bilateralFilter(background_temp_float, 5, 50, 50)
# Generate target heat signatures
targets = []
for _ in range(num_targets):
    target_x = np.random.randint(0, grid_size)
    target_y = np.random.randint(0, grid_size)
    target_temp = np.random.normal(target_temp_mean, target_temp_std, (5, 5))
    targets.append((target_x, target_y, target_temp))
background_temp_smooth = np.zeros_like(background_temp)
for i in range(1, background_temp.shape[0]-1):
    for j in range(1, background_temp.shape[1]-1):
        background_temp_smooth[i, j] = np.mean(background_temp[i-1:i+2, j-1:j+2])
# Add targets to the background grid
for target_x, target_y, target_temp in targets:
    padded_target_temp = np.pad(target_temp, ((0, 0), (0, 1)), mode='constant')  # Add a single column of zeros
    background_temp[target_x-2:target_x+3, target_y-2:target_y+3] += padded_target_temp[:, :5]

# Add noise to the data
noise = np.random.normal(0, 1, (grid_size, grid_size))
background_temp += noise
threshold_temp = 32  # °C

# Apply thresholding to detect targets
target_mask = background_temp > threshold_temp

# Visualize the results
plt.imshow(background_temp, cmap='hot')
plt.title('Original Temperature Grid')
plt.colorbar()
plt.show()

plt.imshow(target_mask, cmap='gray')
plt.title('Thresholded Target Mask')
plt.colorbar()
plt.show()
