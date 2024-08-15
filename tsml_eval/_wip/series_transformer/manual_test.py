from _pla import PLASeriesTransformer
from aeon.datasets import load_electric_devices_segmentation, load_italy_power_demand
from aeon.visualisation import plot_series_with_change_points, plot_series_with_profiles
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler


ts, period_size, true_cps = load_electric_devices_segmentation()
ts = ts.values
ts = ts[:50]

pla = PLASeriesTransformer(max_error=0.02, transformer="swab")
results = pla.fit_transform(ts)

print("Original: ", ts)
print("PLA     : ", results)

plt.subplot(2, 1, 1)  # (rows, columns, subplot_number)
plt.plot(np.arange(len(ts)), ts)
plt.title('Original')
plt.xlabel('x')
plt.ylabel('y1')

# Create the second subplot (lower plot)
plt.subplot(2, 1, 2)  # (rows, columns, subplot_number)
plt.plot(np.arange(len(ts)), results)
plt.title('PLA')
plt.xlabel('x')
plt.ylabel('y2')

# Adjust layout to prevent overlapping
plt.tight_layout()

# Display the plot
plt.show()