import matplotlib.pyplot as plt
import numpy as np

streams = [1, 2, 4, 8]
tiempo_h2d = [0.165888, 0.175424, 0.191424, 0.302176]
tiempo_kernel = [0.100960, 0.108480, 0.104448, 0.137760]
tiempo_d2h = [0.044256, 0.051200, 0.071680, 0.120256]

# graficas
plt.rcParams['figure.figsize'] = (15, 5)
plt.rcParams['font.size'] = 10
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

# Gráfica 1: Tiempo Host->Device
ax1.plot(streams, tiempo_h2d, marker='o', linewidth=2, markersize=8, color='#2E86AB')
ax1.set_xlabel('Cantidad de Streams', fontweight='bold')
ax1.set_ylabel('Tiempo (ms)', fontweight='bold')
ax1.set_title('Tiempo Host->Device', fontweight='bold', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(streams)

# Gráfica 2: Tiempo Kernel
ax2.plot(streams, tiempo_kernel, marker='s', linewidth=2, markersize=8, color='#A23B72')
ax2.set_xlabel('Cantidad de Streams', fontweight='bold')
ax2.set_ylabel('Tiempo (ms)', fontweight='bold')
ax2.set_title('Tiempo Kernel', fontweight='bold', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(streams)

# Gráfica 3: Tiempo Device->Host
ax3.plot(streams, tiempo_d2h, marker='^', linewidth=2, markersize=8, color='#F18F01')
ax3.set_xlabel('Cantidad de Streams', fontweight='bold')
ax3.set_ylabel('Tiempo (ms)', fontweight='bold')
ax3.set_title('Tiempo Device->Host', fontweight='bold', fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.set_xticks(streams)

plt.tight_layout()
plt.savefig('streams_analysis.png', dpi=300, bbox_inches='tight')
plt.show()