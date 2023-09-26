"""The `mill` subpackage is used to specify the beam settings of the ion beam microscope."""
from fibomat.mill.mill import MillBase, Mill, SpecialMill
# from fibomat.mill.ionbeam import IonBeam, GaussBeam

__all__ = ['MillBase', 'Mill', 'SpecialMill']  # 'IonBeam', 'GaussBeam'
