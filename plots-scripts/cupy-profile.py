import numpy as np
import matplotlib.pyplot as plt

# Generate random data for each run size
cpu_profiler = [475.70, 1360.78, 7724.72, 57526.66, 455073.43]
custom_profiler = [211.57119914889336, 1021.3155210018158, 7225.207686424255, 56997.912826538086, 453602.53326416016]
nsize = [1024, 2048, 4096, 8192, 16384]
# Calculate standard deviation for each run size


# Create box plots
plt.ylabel('Elapsed Time Log scaled (microseconds)')
plt.xlabel('Matrix Size')
plt.title('Custom and Python Profiler for Device 0(A100) niterations 100')

# Plot a line connecting the averages
plt.semilogy(nsize, custom_profiler, marker='o', color='green', label='Custom Profiler')
plt.semilogy(nsize, cpu_profiler, marker='o', color='red', label='CPU Profiler')

plt.legend()

plt.show()