from typing import List

from fibomat.units import Q_


class ElyVersion:
    pass


class ElyAxes:
    pass


class ElyGrid:
    pass


class ElyGIS:
    pass


class ElyProbe:
    def __init__(self, name: str, current: Q_, diameter: Q_) -> None:
        if not current.check("[current]"):
            raise ValueError('"current" must be in current units (e.g. A, pA, ...)')

        if not diameter.check("[length]"):
            raise ValueError('"diameter" must be in length units.')

        self.name = name
        self.current = str(current.m_as("A")) + " A"
        self.diameter = str(diameter.m_as("m")) + " m"


class ElyExposure:
    def __init__(
        self, dwell_times_area, dose_area, pixel_spacing, track_spacing, probe, gis
    ):
        self.dwell_times_area = str(dwell_times_area.m_as("s")) + " s"
        self.dose_area = str(dose_area.m_as("C/m**2")) + " C/m&#178;"
        self.pixel_spacing_area = str(pixel_spacing.m_as("m")) + " m"
        self.track_spacing = str(track_spacing.m_as("m")) + " m"

        self.probe = probe
        self.gis = gis


class ElyPolyline:
    def __init__(self, points):
        self.points = " ".join([f"({x:.5f} {y:.5f})" for (x, y) in points])


class ElyArc:
    def __init__(self, cx, cy, r1, r2, a1, a2, exposure):
        self.cx = cx
        self.cy = cy
        self.r1 = r1
        self.r2 = r2
        self.a1 = f"{a1:.5f} deg"
        self.a2 = f"{a2:.5f} deg"

        if exposure:
            self.exposure = exposure
            print(exposure)


class ElyPoint:
    def __init__(self, point, exposure=None):
        self.x = f"{point[0]:.5f}"
        self.y = f"{point[1]:.5f}"

        if exposure:
            self.exposure = exposure
            print(exposure)


class ElyLayer:
    def __init__(self, name: str, fov: float):
        self.name = name
        self.fov = fov
        self.geoms = []


class ElyLayerList:
    def __init__(self, layers: List[ElyLayer]):
        self.layers = layers


class ElyInstanceList:
    pass


class ElyStructure:
    def __init__(self, name: str, layers: List[ElyLayer]):
        self.layers = layers

        self.name = name

        self.version = ElyVersion()
        self.instance_list = ElyInstanceList()


class ElyStructureList:
    def __init__(self, layers: List[ElyLayer]):
        self.structures = [ElyStructure("Structure", layers)]


class ElyFile:
    def __init__(self, name: str, layers: List):
        self.name = name

        self.version = ElyVersion()
        self.axes = ElyAxes()
        self.grid = ElyGrid()

        self.layer_list = ElyLayerList(layers)

        self.structure_list = ElyStructureList(layers)
