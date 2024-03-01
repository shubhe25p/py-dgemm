import numpy as np
import matplotlib.pyplot as plt

# Generate random data for each run size
int8 = [31.254656751127346, 5873.743641199802, 5817.339293706832, 5819.281693943597, 5862.345101221279, 5855.073223899111, 5895.906981159571, 5814.055107187764, 5854.619330525337, 5819.281693943597]
fp32 = [2645.399825701142, 9039.935900410965, 6432.750730302944, 6435.674292628888, 6452.904032126684, 6393.721454791737, 6395.345738335168, 6368.560644029682, 6399.500437012766, 6382.7341316731]
fp64 = [2375.5233336688475, 6700.021098651527, 6627.505805651436, 6404.927728614236, 6383.093772982111, 6429.64735433339, 6542.532875310425, 7284.90767087875, 13071.477151791381, 6320.238603883062]

# Calculate standard deviation for each run size
int8_std = np.std(int8)
fp32_std = np.std(fp32)
fp64_std = np.std(fp64)

# Create box plots
plt.ylabel('Output Variability')
plt.title('Variability of GFLOPS for Different dtypes')

# Calculate averages
int8_avg = np.mean(int8)
fp32_avg = np.mean(fp32)
fp64_avg = np.mean(fp64)

int8_max = np.max(int8)
fp32_max = np.max(fp32)
fp64_max = np.max(fp64)

# Plot a line connecting the averages
plt.plot(int8, marker='o', color='green', label='int8')
plt.plot(fp32, marker='o', color='red', label='fp32')
plt.plot(fp64, marker='o', color='blue', label='fp64')

plt.legend()

plt.show()