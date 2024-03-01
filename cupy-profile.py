import numpy as np
import matplotlib.pyplot as plt

# Generate random data for each run size
profiler_output = [15058.1869, 179517.455, 178871.071, 178871.071]
python_output = [13682.48, 13747.23, 13693.99, 13727.49]
# Calculate standard deviation for each run size


# Create box plots
plt.ylabel('GFLOPS')
plt.title('Comparing Variability of Cupy Profiler and Python time Module')

# Calculate averages
# avg_10 = np.mean(data_10)
# avg_100 = np.mean(data_100)
# avg_500 = np.mean(data_500)
# avg_1000 = np.mean(data_1000)

# max_10 = np.max(data_10)
# max_100 = np.max(data_100)
# max_500 = np.max(data_500)
# max_1000 = np.max(data_1000)

# first_10 = 602.590969508156
# first_100 = 669.7147009207573
# first_500 = 652.9374114462678
# first_1000 = 691.1794217864776

# Plot a line connecting the averages
plt.plot([1, 2, 3, 4], profiler_output, marker='o', color='green', label='Cupy Profiler')
plt.plot([1, 2, 3, 4], python_output, marker='o', color='red', label='Python time Module')

plt.legend()

plt.show()