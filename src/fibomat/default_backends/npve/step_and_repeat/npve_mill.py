from typing import Optional

from fibomat.mill import MillBase
from fibomat.units import TimeQuantity, QuantityType


class NPVEMill(MillBase):
    def __init__(
        self,
        *,
        dwell_time: TimeQuantity,
        repeats: Optional[int] = None,
        dose: Optional[QuantityType] = None,
        **kwargs
    ):
        if (repeats is not None and dose is not None) or (
            repeats is None and dose is None
        ):
            raise ValueError(
                "One of repeats or dose must be given and not both or none of them."
            )

        super().__init__(dwell_time=dwell_time, repeats=repeats, dose=dose, **kwargs)
