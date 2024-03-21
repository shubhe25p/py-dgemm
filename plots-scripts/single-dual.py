import numpy as np
import matplotlib.pyplot as plt

# Set width of bars
barWidth = 0.25

# Set heights of bars
omp16 = [590.01, 591.47, 605.39, 604.69, 610.79, 596.25, 605.27]
omp32 = [1086.08, 1091.09, 1101.96, 1094.14, 1112.95, 1096.02, 1095.52]
omp64 = [1412.38,1400.78,1417.57,1447.36,1454.03,1413.99,1460.28]
omp128 = [1399.95,1446.77,1369.61,1345.33,1388.31,1363.37,1409.40]

## Set position of bars on X axis
#r1 = 1.25 * np.arange(len(omp16))
#r2 = [x + barWidth for x in r1]
#r3 = [x + barWidth for x in r2]
#r4 = [x + barWidth for x in r3]
#
#new_bars1 = omp16 - np.mean(omp16)
#new_bars2 = omp32 - np.mean(omp32)
#new_bars3 = omp64 - np.mean(omp64)
#new_bars4 = omp128 - np.mean(omp128)
#
#fig, ax = plt.subplots()
#
#ax.axhline(np.mean(omp16), color='r', linestyle='--', label='mean-omp16')
#ax.axhline(np.mean(omp32), color='r', linestyle='--', label='mean-omp32')
#ax.axhline(np.mean(omp64), color='b', linestyle='--', label='mean-omp64')
#ax.axhline(np.mean(omp128), color='r', linestyle='--', label='mean-omp128')
#
#plt.plot(r1, omp16, 'o', label='omp16')
#plt.plot(r2, omp32, 'o', label='omp32')
#plt.plot(r3, omp64, 'o', label='omp64')
#plt.plot(r4, omp128, 'o', label='omp128')
#
# Add xticks on the middle of the group bars
#plt.xlabel('NUMA Nodes', fontweight='bold')
#plt.ylabel('GFLOPs', fontweight='bold')
#plt.title('AMD EPYC 7763 2x CPU run, nsize 4096: (metric: avg 100 runs, 100 times warmup)')
#r1 = [r+1 for r in range(len(omp16))]
#plt.xticks(r1, ['N0-N1', 'N0-N2', 'N0-N3', 'N0-N4', 'N0-N5', 'N0-N6', 'N0-N7'])
#
# Create legend & Show graphic
#plt.legend()
#plt.show()

#-# Total Plot
omp16M = np.mean(omp16)
omp32M = np.mean(omp32)
omp64M = np.mean(omp64)
omp128M = np.mean(omp128)

fig, ax = plt.subplots()
aBar,bBar,cBar,dBar = plt.bar([1,2,3,4],[omp16M,omp32M,omp64M,omp128M])
aBar.set_facecolor('goldenrod')
bBar.set_facecolor('seagreen')
cBar.set_facecolor('royalblue')
dBar.set_facecolor('orchid')
ax.title.set_text('AMD EPYC 7763 2x run, N=4096: (avg 100 runs, 100 times warmup)')

plt.ylabel('GFLOPs', fontweight='bold')
plt.xlabel('OpenMP Threads', fontweight='bold')
plt.xticks([1,2,3,4],['16','32','64','128'])
plt.ylim(0,1600)

plt.show()
plt.close()

#-# Differences Plot
omp16D = (omp16 - omp16M) / omp16M * 100
omp32D = (omp32 - omp32M) / omp32M * 100
omp64D = (omp64 - omp64M) / omp64M * 100
omp128D = (omp128 - omp128M) / omp128M * 100

fig, ax = plt.subplots()
plt.plot(omp16D,'*:',color='goldenrod',label='OMP = 16')
plt.plot(omp32D,'*:',color='seagreen',label='OMP = 32')
plt.plot(omp64D,'*:',color='royalblue',label='OMP = 64')
plt.plot(omp128D,'*:',color='orchid',label='OMP = 128')
ax.axhline(0, color='k', linestyle='--')

plt.ylabel('% Difference from Mean', fontweight='bold')
plt.xlabel('Numa Node (partnered with N0)', fontweight='bold')
plt.xticks([0,1,2,3,4,5,6],['N1','N2','N3','N4','N5','N6','N7'])
plt.title('AMD EPYC 7763 2x run, N=4096: (avg 100 runs, 100 times warmup)')
plt.legend(title='OpenMP Threads')
plt.show()