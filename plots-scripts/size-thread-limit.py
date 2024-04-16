import numpy as np
import matplotlib.pyplot as plt

# Set width of bars
barWidth = 0.25

# Set heights of bars
bars1 = [3.4,15.2,152.8,713.6,1203.9,1950.7,1040.4,1524.8,1555.7,2339.8,3068.4,3158.9,3181.2,3274.1,3233.1,3309.0,3391.7,2687.2]
size = ['32', '64', '128', '256', '512', '1024', '2048', '4096', '8192', '16384', '32768', '35K', '40K', '45K', '50K', '55K', '60K', '65536']
print(len(bars1))
print(len(size))
# Set position of bars on X axis
r1 = np.arange(len(bars1))
print(r1)
# Make the plot
plt.bar(r1, bars1, color='#e9724d', edgecolor='white')

# Add xticks on the middle of the group bars
plt.xlabel('Size of matrix', fontweight='bold', fontsize=15)
plt.ylabel('GFLOPs', fontweight='bold', fontsize=15)
plt.title('Performance of DGEMM')
r1 = [r + 0.75 for r in range(len(bars1))]
# plt.xticks(r1, size)

# Create legend & Show graphic
plt.legend()
plt.show()

plt.savefig("sample.png")