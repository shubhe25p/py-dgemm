import numpy as np
import matplotlib.pyplot as plt

# Set width of bars
barWidth = 0.25

# Set heights of bars
bars1 = [1065.6,1126.3,1500.7,1489.9,1997.2]
bars2 = [1136.8,953.6,1457.6,1572.3,2248.9]
bars3 = [1165.9,968.1,1377.1,1588.8,2208.0]
bar4 = [1121.4,1029.6,1352.8,1595.2,2246.0]
bar5 = [1181.0,982.7,1404.5,1585.6,2306.6]
bar6 = [1140.5,983.0,1379.4,1560.2,2210.8]

# Set position of bars on X axis
r1 = 1.75 * np.arange(len(bars1))
r2 = [x +  barWidth for x in r1]
r3 = [x +  barWidth for x in r2]
r4 = [x +  barWidth for x in r3]
r5 = [x +  barWidth for x in r4]
r6 = [x +  barWidth for x in r5]

# Make the plot
plt.bar(r1, bars1, color='#e9724d', width=barWidth, edgecolor='white', label='omp64')
plt.bar(r2, bars2, color='#d6d727', width=barWidth, edgecolor='white', label='omp128')
plt.bar(r3, bars3, color='#008000', width=barWidth, edgecolor='white', label='omp256')
plt.bar(r4, bar4, color='orchid', width=barWidth, edgecolor='white', label='omp512')
plt.bar(r5, bar5, color='#236192', width=barWidth, edgecolor='white', label='omp1024')
plt.bar(r6, bar6, color='#fea601', width=barWidth, edgecolor='white', label='omp2048')

# Add xticks on the middle of the group bars
plt.xlabel('Size of matrix', fontweight='bold', fontsize=15)
plt.ylabel('GFLOPs', fontweight='bold', fontsize=15)
plt.title('Performance of DGEMM, 2xAMD EPYC 7763 256 logical cores, 10 iterations', fontdict={'fontsize': 15})
r1 = [r + 0.75 for r in range(len(bars1))]
r1[1] = r1[1] + 0.5
r1[2] = r1[2] + 1.25
r1[3] = r1[3] + 1.75
r1[4] = r1[4] + 2.75
plt.xticks(r1, ['1024', '2048', '4096', '8192', '16384'])

# Create legend & Show graphic
plt.legend()
plt.show()