'''
Author: TX-Leo
Mail: tx.leo.wz@gmail.com
Date: 2024-07-01 08:09:50
Version: v1
File: 
Brief: 
'''
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt

# File paths
input_tif = r"D:\Desktop\earthlab.tif"
output_tif = r"D:\Desktop\earthlab_cropped.tif"

# Cropping bounds (in pixel coordinates)
left, bottom, right, top = 1500, 1500, 2000, 2200 

with rasterio.open(input_tif) as src:
    # Construct a rasterio Window object 
    window = Window(left, bottom, right - left, top - bottom)

    # Read the data within the window
    cropped_data = src.read(window=window)

    # Update metadata for the cropped image
    out_meta = src.meta.copy()
    out_meta.update({
        "height": window.height,
        "width": window.width,
        "transform": rasterio.windows.transform(window, src.transform)
    })

    # Write the cropped data to a new GeoTIFF file
    with rasterio.open(output_tif, "w", **out_meta) as dest:
        dest.write(cropped_data)


# --- Visualization --- 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6)) 

# Plot the original raster
with rasterio.open(input_tif) as src:
    raster_data = src.read(1) # Read the first band
    img1 = ax1.imshow(raster_data, cmap='gray') 
    ax1.set_title("Original Raster")
    # Add axis labels and ticks
    ax1.set_xlabel("Column #")
    ax1.set_ylabel("Row #")

# Plot the cropped raster
with rasterio.open(output_tif) as src:
    raster_data = src.read(1) # Read the first band
    img2 = ax2.imshow(raster_data, cmap='gray') 
    ax2.set_title("Cropped Raster")
    # Add axis labels and ticks
    ax2.set_xlabel("Column #")
    ax2.set_ylabel("Row #")

# Add colorbars
plt.colorbar(img1, ax=ax1, label="Pixel Value")
plt.colorbar(img2, ax=ax2, label="Pixel Value")

plt.show()
plt.savefig(r"D:\Desktop\vis.png")