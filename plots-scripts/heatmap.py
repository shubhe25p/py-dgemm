import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Creating a 10x10 array
data = [[590.01, 591.47, 605.39, 604.69, 610.79, 596.25, 605.27],
        [1086.08, 1091.09, 1101.96, 1094.14, 1112.95, 1096.02, 1095.52],
        [1412.38,1400.78,1417.57,1447.36, 1454.03,1413.99,1460.28],
        [1399.95,1446.77,1369.61,1345.33,1388.31,1363.37,1409.40]]
# Creating a heatmap using imshow()
plt.subplots(figsize=(10, 5))
hmap = sns.heatmap( data, linewidth = 1)
plt.title( "Same Socket vs Dual Socket GFLOPS Comparison" )
hmap.set_xticklabels(['N0-N1', 'N0-N2', 'N0-N3', 'N0-N4', 'N0-N5', 'N0-N6', 'N0-N7'])
hmap.set_yticklabels(['OMP16', 'OMP32', 'OMP64', 'OMP128'])
plt.show()