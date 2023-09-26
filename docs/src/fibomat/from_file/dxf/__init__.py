from typing import Union, List
import pathlib
import math

import numpy as np

import ezdxf
import ezdxf.entities

from fibomat.shapes import Shape, Polyline, Polygon, Line, Spot, Circle, Ellipse, Rect
from fibomat.utils import PathLike


def _is_rect(vertices):
    if len(vertices) == 4:
        # check if the diagonals are equal
        if math.isclose(np.linalg.norm(vertices[0]-vertices[2]),
                        np.linalg.norm(vertices[1]-vertices[3])):
            return True
    return False


def _rect_from_vertices(vertices):

    # find the vertex with the minimum y value to get a base site of the rect.
    i_min = np.where(vertices[:, 1] == vertices[:, 1].min())[0]

    if len(i_min) == 1:
        # we found one, now check, if the left or right vertex will result in a
        # smaller rotation of the base site compared to x axis.
        base_1 = vertices[(i_min[0]+1) % 4] - vertices[i_min[0]]
        base_2 = vertices[i_min[0]] - vertices[(i_min[0]-1) % 4]

        if np.abs(np.arctan2(base_1[1], base_1[0])) < np.abs(np.arctan2(base_2[1], base_2[0])):
            base_start_vertex_i = i_min[0]
            base_end_vertex_i = (i_min[0]+1) % 4

            theta = np.arctan2(base_1[1], base_1[0])
        else:
            base_start_vertex_i = (i_min[0]-1) % 4
            base_end_vertex_i = i_min[0]

            theta = np.arctan2(base_2[1], base_2[0])

    elif len(i_min) == 2:
        # we got two, the base site is parallel to x axis
        base_start_vertex_i = i_min[0]
        base_end_vertex_i = i_min[1]

        if vertices[base_start_vertex_i, 0] > vertices[base_end_vertex_i, 0]:
            base_start_vertex_i, base_end_vertex_i = base_end_vertex_i, base_start_vertex_i

        base = vertices[base_end_vertex_i] - vertices[base_start_vertex_i]
        theta = np.arctan2(base[1], base[0])
    else:
        raise RuntimeError('Something went horribly wrong.')

    width = np.linalg.norm(vertices[base_start_vertex_i]-vertices[base_end_vertex_i])
    height = np.linalg.norm(vertices[base_end_vertex_i]-vertices[(base_end_vertex_i+1) % 4])

    center = np.mean(vertices, axis=0)

    return Rect(width=width, height=height, theta=theta, center=center)


# TODO: let vertices_raw be a generator object
def _polyline(ocs, vertices_raw, closed):
    vertices = np.empty(shape=(len(vertices_raw), 2))
    for i, vertex in enumerate(vertices_raw):
        vertices[i][0] = ocs.to_wcs(vertex)[0]
        vertices[i][1] = ocs.to_wcs(vertex)[1]

    if closed:
        if _is_rect(vertices):
            return _rect_from_vertices(vertices)
        return Polygon(vertices)

    # it could be possible that start and end are equal but it is not marked as
    # closed.
    if np.allclose(vertices[0], vertices[-1]):
        cropped_vertices = vertices[:-1]
        if _is_rect(cropped_vertices):
            return _rect_from_vertices(cropped_vertices)
        return Polygon(cropped_vertices)

    return Polyline(vertices)


def _ellipse(major, minor, center):
    major = np.array(major)
    minor = np.array(minor)

    angle_major = np.abs(np.arctan2(major[1], major[0]))

    if angle_major > np.pi/2:
        major *= -1.
        angle_major -= np.pi

    angle_minor = np.abs(np.arctan2(minor[1], minor[0]))
    if angle_minor > np.pi / 2:
        angle_minor *= -1.
        angle_major -= np.pi

    # if angle_major > np.pi/2:
    #     angle_major -= np.pi / 2
    #
    # if angle_minor > np.pi / 2:
    #     angle_minor -= np.pi / 2

    print(angle_major, angle_minor)

    if angle_major <= angle_minor:
        a = np.linalg.norm(major)
        b = np.linalg.norm(minor)

        theta = np.arctan2(major[1], major[0])

    else:
        a = np.linalg.norm(minor)
        b = np.linalg.norm(major)

        theta = np.arctan2(minor[1], minor[0])

    if theta > np.pi / 2:
        theta -= np.pi
    elif theta < -np.pi / 2:
        theta += np.pi

    return Ellipse(a, b, theta, center)


def shapes_from_dxf(file_path: PathLike) -> List[Shape]:
    """
    Reads a dxf file an convert the contained entities (shapes) to fibomat shapes.

    .. note:: z components of all entities are ignored.

    Args:
        file_path (pathlib.Path): path to file

    Returns:
        List[Shape]
    """
    doc = ezdxf.readfile(str(file_path))

    shapes = []

    # we try to get all objects from modelspace
    for obj in doc.modelspace():
        ocs = obj.ocs()
        dxftype = obj.dxftype()

        if dxftype == 'POINT':
            spot: ezdxf.entities.Point = obj
            shapes.append(Spot(spot.dxf.location[:2]))
        elif dxftype == 'LINE':
            line: ezdxf.entities.Line = obj
            shapes.append(Line(line.dxf.start[:2], line.dxf.end[:2]))
        elif dxftype == 'POLYLINE':
            poly: ezdxf.entities.Polyline = obj

            # maybe these to clauses are equivalent
            if not poly.is_2d_polyline or not poly.get_mode() == 'AcDb2dPolyline':
                raise RuntimeError('Polyline is not 2d.')

            if poly.has_arc:
                raise RuntimeError('Arcs are not supported in polyline.')

            vertices = list(poly.points())
            shapes.append(_polyline(ocs, vertices, poly.is_closed))
        elif dxftype == 'LWPOLYLINE':
            poly: ezdxf.entities.LWPolyline = obj

            if poly.has_arc:
                raise RuntimeError('Arcs are not supported in polyline.')

            vertices = list(poly.vertices())
            shapes.append(_polyline(ocs, vertices, poly.closed))
        elif dxftype == 'CIRCLE':
            circle: ezdxf.entities.Circle = obj
            shapes.append(
                Circle(
                    r=circle.dxf.radius,
                    center=ocs.to_wcs(circle.dxf.center)[:2])
            )
        elif dxftype == 'ELLIPSE':
            ellipse: ezdxf.entities.Ellipse = obj
            if not math.isclose(ellipse.dxf.start_param, 0.) or not math.isclose(ellipse.dxf.end_param, 2.* np.pi):
                raise RuntimeError("Elliptical arcs are not supported.")

            shapes.append(_ellipse(ellipse.dxf.major_axis[:2], ellipse.minor_axis[:2], ellipse.dxf.center[:2]))
        else:
            raise RuntimeError(
                f'Encountered (currently) unsupported shape {obj.dxftype()}'
            )

    return shapes
