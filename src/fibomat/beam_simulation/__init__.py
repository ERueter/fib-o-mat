import sys
import argparse
import configparser

import numpy as np

try:
    from PyQt5 import QtWidgets
    from fibomat.beam_simulation.window import AnimationWindow
except ModuleNotFoundError as error:
    raise RuntimeError(
        "You need to install the package with 'gui' as optional dependency to be able to use the beam-simulation tool."
    ) from error

from fibomat.linalg import BoundingBox


def _load_and_prepare(file: str):

    with open(file, 'r') as fp:
        info_lines = []
        while 'Points' not in (line := fp.readline()):
            info_lines.append(line)

        points = np.loadtxt(fp, delimiter=' ')

    info = configparser.ConfigParser()
    info.read_string(''.join(info_lines))
    length_unit = info['Info']['length_unit']

    # points[:, 2] = 0.
    bbox = BoundingBox.from_points(points[:, :2])
    points[:, :2] -= bbox.center

    # flip at x axis because the canvas system is defined like this !?
    points[:, 1] *= -1

    return points, length_unit


def run():
    parser = argparse.ArgumentParser(description='Ion beam simulation.')
    parser.add_argument('filename', metavar='filename', type=str, nargs=1,
                        help='filename of spot list file')

    args = parser.parse_args()
    file = args.filename[0]

    appQt = QtWidgets.QApplication(sys.argv)
    win = AnimationWindow(*_load_and_prepare(file))
    win.show()
    appQt.exec_()
