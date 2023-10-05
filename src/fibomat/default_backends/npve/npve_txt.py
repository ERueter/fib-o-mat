from typing import Dict, Any
import warnings

import numpy as np

from fibomat import U_, Q_
from fibomat.default_backends import SpotListBackend
from fibomat import utils


class NPVETxt(SpotListBackend):
    name = "NPVEtxt"

    def __init__(self, description=None):
        def _save_impl(
            filename: utils.PathLike,
            dwell_points: np.ndarray,
            parameters: Dict[str, Any],
        ):
            if "base_dwell_time" not in parameters or not parameters["base_dwell_time"]:
                raise RuntimeError

            warnings.warn(
                "The base dwell time is set to 0.1 us by default. Open a bug report if you need a different base dwell time."
            )

            fov = max(parameters["fov"].width, parameters["fov"].height)

            with open(filename, "w") as fp:
                # header.write(fp)
                # fp.write('[Points]\n')
                fp.write("NPVE DEFLECTION LIST\r\n")
                fp.write("UNITS=MICRONS\r\n")
                fp.write("DWELL=0.1\r\n")
                fp.write(f"FOV={fov}\r\n")
                fp.write("START\r\n")
                np.savetxt(fp, dwell_points, "%.5f %.5f %d", newline="\r\n")
                fp.write("END\r\n")

        super().__init__(
            save_impl=_save_impl,
            base_dwell_time=Q_("0.1 µs"),
            length_unit=U_("µm"),
            time_unit=U_("µs"),
        )
