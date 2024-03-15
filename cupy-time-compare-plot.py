py=[18923.20,19078.48,19043.23,18922.75]
cpy=[1503614.95, 1494282.63, 1491186.77, 18900.3395]
cpyt=[7339.14, 94.97, 95.21, 95.50]
custom=[94.04, 94.49, 94.78, 7274.67]

import matplotlib.pyplot as plt


plt.plot(cpyt, marker='o', color='b', label='CuPy Profiler')
plt.plot(custom, marker='o', color='r', label='Custom Profiler')
# plt.semilogy(cpyt, marker='o', color='b', label='CuPy Profiler')
# plt.semilogy(custom, marker='o', color='r', label='Custom Profiler')
plt.xticks([0, 1, 2, 3], ['GPU0', 'GPU1', 'GPU2', 'GPU3'])
plt.yscale("log")
# plt.ticklabel_format(style='plain', axis='y')
plt.title("Elapsed time comparison between two profilers")
plt.legend()
plt.show()