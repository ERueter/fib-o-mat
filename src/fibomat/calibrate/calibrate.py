from fibomat import Sample, U_, Mill, Q_, DDDMill, SILMill, MatrixMill, MillBase
from fibomat import shapes, raster_styles
from fibomat.units import QuantityType
from fibomat.default_backends.fei import FEIStreamFile
import os
import numpy as np
from typing import Callable

def calibrate(rasterstyle: raster_styles.RasterStyle) -> Callable[[float], int]:
    print("In this calibration we will determine how much material is removed per given time.")
    print("Please set the microscope to the current and voltage you are intending to use. See documentation for examples.")

    s = Sample()
    site = s.create_site(
        dim_position=(0, 0) * U_('µm'), dim_fov=(20, 20) * U_('µm')
    )

    mill = Mill(10*Q_("µs"), repeats=1)
    circ1 = shapes.Circle(r=2.5, center=(-6,0))
    circ2 = shapes.Circle(r=2.5, center=(0,0))
    circ3 = shapes.Circle(r=2.5, center=(6,0))

    os.mkdir("calibration-files")

    site.create_pattern(
        dim_shape=circ1 * U_('µm'),
        mill=mill,
        raster_style=rasterstyle
    )

    exported10 = s.export(FEIStreamFile, n_rep=10, margin=0.76) 
    exported10.save('calibration-files/10repeats')
    del exported10

    site.empty_site()

    site.create_pattern(
        dim_shape=circ2 * U_('µm'),
        mill=mill,
        raster_style=rasterstyle
    )

    exported100 = s.export(FEIStreamFile, n_rep=100, margin=0.76) 
    exported100.save('calibration-files/100repeats')
    del exported100

    site.empty_site()

    site.create_pattern(
        dim_shape=circ3 * U_('µm'),
        mill=mill,
        raster_style=rasterstyle
    )

    exported250 = s.export(FEIStreamFile, n_rep=250, margin=0.76) 
    exported250.save('calibration-files/250repeats')
    del exported250

    print("Three calibration patterns were generated and saved in the folder 'calibration-files'.")
    print("Please mill these three files and measure the depth of the resulting circles.")

    depth10 = float(input("Enter depth of the 10-repeat circle in µm: "))
    depth100 = float(input("Enter depth of the 100-repeat circle in µm: "))
    depth250 = float(input("Enter depth of the 250-repeat circle in µm: "))

    # Assume linear relationship through origin: depth = a * repeats (b=0)
    repeats = np.array([10, 100, 250])
    depths = np.array([depth10, depth100, depth250])
    
    # Calculate slope a for line through origin
    a = np.sum(depths * repeats) / np.sum(repeats**2)
    b = 0  # Forced to pass through (0,0)
    
    # Calculate R² to check goodness of fit (for regression through origin)
    predicted = a * repeats
    ss_res = np.sum((depths - predicted)**2)
    ss_tot = np.sum(depths**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    print(f"Linear relationship through origin: depth = {a:.6f} * repeats [µm]")
    print(f"Goodness of fit (R²): {r_squared:.4f}")
    
    if r_squared < 0.9:
        print("Warning: The linearity assumption fits poorly (R² < 0.9). Consider using a non-linear model or checking your measurements.")
    
    # Function to calculate repeats for a given depth TODO add units
    def repeats_for_depth(desired_depth):
        if a == 0:
            raise ValueError("Slope a is zero, cannot determine repeats.")
        return int(desired_depth / a)

    return repeats_for_depth