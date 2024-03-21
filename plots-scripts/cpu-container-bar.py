
import numpy as np
import matplotlib.pyplot as plt

# Set width of bars
barWidth = 0.25

# Set heights of bars
bars1 = [897.6,853.0,912.4,846.7,890.4,738.3]
bars2 = [1500.7,1457.6,1377.1,1352.8,1404.5,1379.4]

# Set position of bars on X axis
r1 = np.arange(len(bars1))
r2 = [x +  barWidth for x in r1]

# Make the plot
plt.bar(r1, bars1, color='#e9724d', width=barWidth, edgecolor='white', label='Podman HPC')
plt.bar(r2, bars2, color='#d6d727', width=barWidth, edgecolor='white', label='Bare Metal')

# Add xticks on the middle of the group bars
plt.xlabel('OpenMP Threads', fontweight='bold')
plt.ylabel('GFLOPs', fontweight='bold')
plt.title('Performance of DGEMM, 2xAMD EPYC 7763, 100 iterations, N=4096')
r1 = [r for r in range(len(bars1))]
# r1[1] = r1[1] + 0.5
# r1[2] = r1[2] + 1.25
# r1[3] = r1[3] + 1.75
# r1[4] = r1[4] + 2.25
plt.xticks(r1, ['OMP64', 'OMP128', 'OMP256', 'OMP512', 'OMP1024', 'OMP2048'])

# Create legend & Show graphic
plt.legend()
plt.show()