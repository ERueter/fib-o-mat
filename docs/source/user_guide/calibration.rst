Calibration
===========

.. _user_guide/calibration:

The calibration feature allows you to experimentally determine the milling rate of your ion beam setup,
enabling accurate prediction of milling depth for a given number of repeats.

Why calibration is needed
--------------------------

Milling depth depends on many hardware-specific factors:

- Ion beam current
- Ion energy (accelerating voltage)
- Material properties
- Rasterization style and pitch
- Dwell time per spot

Rather than trying to predict depth theoretically, the calibration routine measures how much material
is removed for a known set of process parameters. From this, you can then calculate the number of
stream file repeats needed to achieve a desired depth.

Calibration workflow
--------------------

The calibration process consists of two steps through which the routine will guide you:

1. **Generate calibration patterns** — Run :func:`~fibomat.calibrate.calibrate` to create three
   test patterns with different numbers of stream file repeats (10, 100, 250).

2. **Measure depths and enter values** — Mill the patterns on the microscope, measure the resulting
   depths, and input them when prompted.

From these measurements, a linear calibration curve is fitted and stored, providing a function
to convert desired depth to the required number of stream file repeats.

Quick start example
-------------------

::

    from fibomat.calibrate import calibrate
    from fibomat import raster_styles, U_

    # Define the rasterization style you plan to use
    spiral_style = raster_styles.two_d.Spiral(
        pitch=20 * U_('nm'),
        spiral_pitch=20 * U_('nm'),
        scan_sequence=raster_styles.ScanSequence.CONSECUTIVE,
        direction="out-in"
    )

    # Run calibration (generates patterns and waits for user input)
    a, repeats_for_depth = calibrate(rasterstyle=spiral_style)

    # a is the milling rate: depth = a * (n_rep * mill_repeats) [µm]
    print(f"Milling rate: {a:.6f} µm per repeat")

    # Use the function to calculate repeats for a desired depth
    from fibomat import Q_
    target_depth = 7.3 * Q_('µm')
    n_repeats = repeats_for_depth(target_depth)
    print(f"Need {n_repeats} stream file repeats to reach {target_depth}")

Function reference
------------------

.. py:function:: calibrate(rasterstyle, mill_repeats=1, max_dwelltime=10 µs)

    Calibrate the milling rate experimentally.

    The function generates three circular test patterns with ``n_rep`` values of 10, 100, and 250.
    These are saved to a ``calibration-files/`` folder for you to mill on the microscope.

    After milling, you are prompted to enter the measured depths. A linear calibration curve
    (passing through the origin) is fitted to determine the milling rate.

    **Parameters:**

    - **rasterstyle** (`RasterStyle`) — The rasterization style to use for the calibration patterns.
      This should match the style you plan to use in actual experiments, as the milling rate
      depends on pitch and scan method.

    - **mill_repeats** (`int`, default=1) — The number of repeats each ``Mill`` object performs
      per stream file repeat. Only relevant if using ``DDDMill`` or custom mills with repeats > 1.

    - **max_dwelltime** (`QuantityType`, default=10 µs) — The dwell time per spot for the calibration
      patterns. Higher dwell times lead to deeper milling but longer acquisition times.

    **Returns:**

    A tuple ``(a, repeats_for_depth)`` where:

    - **a** (`float`) — The milling rate constant (depth in µm per repeat).
      Milling follows: ``depth [µm] = a * (n_rep * mill_repeats)``

    - **repeats_for_depth** (`Callable`) — A function that takes a desired depth as a
      `QuantityType` and returns the required number of stream file repeats (integer).

    **Raises:**

    - `ValueError` — If the calculated slope is zero (no material removed).
    - Error, if the folder where the calibration patterns are stored already exist, that is, if you call the method twice without removing the generated folder after the first time.
    - Prints a warning if the fit quality (R²) is below 0.9, suggesting non-linearity.

Typical usage in experiments
----------------------------

Once calibrated, you can integrate depth control into your patterning workflow:

::

    from fibomat import Sample, U_, Mill, Q_, shapes, raster_styles
    from fibomat.default_backends.fei import FEIStreamFile
    from fibomat.calibrate import calibrate

    # Calibrate once per unique (rasterization style, current, voltage) combination
    spiral_style = raster_styles.two_d.Spiral(
        pitch=20 * U_('nm'),
        spiral_pitch=20 * U_('nm')
    )
    a, repeats_for_depth = calibrate(rasterstyle=spiral_style)

    # Now use repeats_for_depth in actual experiments
    s = Sample()
    site = s.create_site(dim_position=(0, 0) * U_('µm'), dim_fov=(20, 20) * U_('µm'))

    mill = Mill(dwell_time=Q_('10 microsecond'), repeats=1)
    shape = shapes.Circle(r=5, center=(0, 0))

    site.create_pattern(
        dim_shape=shape * U_('µm'),
        mill=mill,
        raster_style=spiral_style
    )

    target_depth = 2.5 * Q_('µm')
    n_rep = repeats_for_depth(target_depth)
    print(f"Using {n_rep} repeats to achieve {target_depth}")

    exported = s.export(FEIStreamFile, n_rep=n_rep, margin=0.76)
    exported.save('my_pattern.str')

Important notes
---------------

**Linearity assumption**

The calibration assumes a linear relationship between repeats and depth. If R² < 0.9, the linearity assumption may not hold; consider:

- Using a smaller target depth
- Checking your measurements for accuracy
- Writing custom non-linear models
Note that angle depended sputter yield is not integrated in this model, so it will probably fail for steep profiles.

**Recalibration**

Recalibrate whenever you change:

- Rasterization style (pitch, scan sequence)
- Ion beam current
- Ion energy (accelerating voltage)
- Material

**Units**

Depths are always returned in **µm**. The :func:`repeats_for_depth` function accepts any
`QuantityType` with length dimensions and automatically converts to µm.

**Mill repeats vs stream file repeats**

- *Stream file repeats* (`n_rep` in the export) — How many times the microscope executes the stream file.
- *Mill repeats* (`mill_repeats` parameter in `calibrate()`) — How many times each ``Mill`` repeats internally (normally, this is not useful if the site contains only a single shape).

The calibration accounts for both: ``depth = a * (n_rep * mill_repeats)``
