import numpy as np
import matplotlib.pyplot as plt

# Set width of bars
barWidth = 0.25

# Set heights of bars
bars1 = [51.46, 51.21, 52.74, 6.14, 53.45]

# Set position of bars on X axis
r1 = np.arange(len(bars1))
fig, ax = plt.subplots()

# Add a dotted horizontal line at y = 0.5
ax.axhline(bars1[0], color='r', linestyle='--')
ax.axhline(bars1[4], color='r', linestyle='--')

# Make the plot
ax.plot(r1, bars1, 'o')

# Add xticks on the middle of the group bars
plt.xlabel('BLAS routines', fontweight='bold')
plt.ylabel('GFLOPs', fontweight='bold')
plt.title('2x AMD EPYC 7763 CPU nsize 4096: (metric: avg 10 runs, no warmups)')
plt.xticks(r1, ['MKL', 'OpenBLAS', 'BLIS', 'NetLib', 'Cray-Python'])

# Create legend & Show graphic
plt.show()