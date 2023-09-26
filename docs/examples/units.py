from fibomat import Vector, U_, Q_
from fibomat.linalg import DimVector
from fibomat.units import scale_factor, scale_to


length_unit = U_('µm')
dose_unit = U_('ions / nm**2')
print(length_unit, dose_unit)

length = Q_('1 nm')
dose = Q_('10 ions / nm**2')
another_length = 10 * length_unit
print(length, dose, another_length)

# three version to create a dimensioned vector
dim_vector = (3, 4) * U_('µm')
dim_vector2 = Vector(3, 4) * U_('µm')
dim_vector3 = DimVector(3 * U_('µm'), 4 * U_('µm'))
print(dim_vector, dim_vector2, dim_vector3)

length_in_um = scale_to(U_('µm'), length)  # NOTE: length_in_um is a float now and NOT a quantity anymore
scale_factor = scale_factor(U_('µm'), U_('nm'))  # scale factor (float) to scale from nm to µm
print(length_in_um, scale_factor)

