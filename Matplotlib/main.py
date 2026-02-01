import numpy as np
import matplotlib.pyplot as plt

x = np.array([2023, 2024, 2025, 2026])
y = np.array([515, 205, 100, 240])

plt.title("Learning Trend", fontsize=20, 
          family="serif", 
          fontweight="bold", 
          color="#5174EA")

plt.xlabel('Year', fontsize=12)
plt.ylabel('No. of Students', fontsize=12)

plt.plot(x, y, marker='o', 
         markersize=10, 
         markerfacecolor='#1cd3fc', 
         linewidth=2)

plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()          # ← this line is actually optional with %matplotlib inline