"""Provide the abstract class :class:`IonBeam` and a implementation :class:`GaussBeam`."""
from typing import Union, overload, Tuple, List
import abc

import numpy as np

from fibomat.units import UnitType, QuantityType, U_, scale_factor
from fibomat.linalg import VectorLike, Vector, VectorValueError


class IonBeam(abc.ABC):
    """Base class to model an ion beam."""

    def __init__(self, beam_cutoff: int):
        """
        Args:
            beam_cutoff (int): beam_cutoff * std is used as cut-off length for flux calculations.

        Raises:
            ValueError: Raised if beam_cutoff < 1.
        """
        beam_cutoff = int(beam_cutoff)

        if beam_cutoff < 1:
            raise ValueError('beam_cutoff must not be smaller than 1.')

        self._beam_cutoff = beam_cutoff

    @abc.abstractmethod
    def __repr__(self):
        pass

    @property
    @abc.abstractmethod
    def std(self) -> QuantityType:
        """Standard deviation of ion beam distribution.

        Access:
            get

        Returns:
            QuantityType
        """

    @overload
    def flux_at(
        self,
        position: VectorLike, beam_maxima: VectorLike,
        position_unit: UnitType
    ) -> QuantityType: ...

    @overload
    def flux_at(
        self,
        position: VectorLike, beam_maxima: List[VectorLike],
        position_unit: UnitType
    ) -> Tuple[np.ndarray, QuantityType]: ...

    @abc.abstractmethod
    def flux_at(
        self,
        position: VectorLike, beam_maxima: Union[List[VectorLike], VectorLike],
        position_unit: UnitType
    ) -> Union[Tuple[np.ndarray, UnitType], QuantityType]:
        """Calculate the ion flux at `position` if the ion beam maximus is at beam_maximum.

        Args:
            position (VectorLike): position at which the flux is calculated.
            beam_maxima (VectorArrayLike, VectorLike): position of beam maximum/maxima
            position_unit (UnitType): length unit of positions and beam_maximum

        Returns:
            Union[QuantityType, Tuple[np.ndarray, QuantityType]]:
                QuantityType is returned if position is VectorLike and Tuple[np.ndarray, QuantityType] if positions is
                VectorArrayLike.
        """

    def nominal_flux_per_spot(self) -> QuantityType:
        """Calculate the ion flux of a single, isolated spot.

        Returns:
            QuantityType
        """
        flux, unit = self.flux_at((0, 0), (0, 0), U_('µm'))
        return flux * unit

    def nominal_flux_per_spot_on_line(self, pitch: QuantityType) -> QuantityType:
        """Calculate the ion flux of a spot on line with pitch `pitch`.

        Args:
            pitch (QuantityType): pitch on line

        Returns:
            QuantityType
        """
        pitch = pitch.to('µm')

        n_considered_spots = int((self.std * self._beam_cutoff / pitch).to_reduced_units().m)

        points = np.c_[
            pitch.m * np.arange(-n_considered_spots, n_considered_spots + 1),
            np.zeros(2*n_considered_spots + 1)
        ]

        fluxes, flux_unit = self.flux_at((0, 0), points, pitch.units)

        return np.sum(fluxes) * flux_unit

    def nominal_flux_per_spot_in_rect(self, pitch_x: QuantityType, pitch_y: QuantityType) -> QuantityType:
        """Calculate the ion flux of a spot on rectangular grid with pitches `pitch_x` and `pitch_y`.

        Args:
            pitch_x (QuantityType): pitch in x direction
            pitch_y (QuantityType): pitch in y direction

        Returns:
            QuantityType
        """
        pitch_x = pitch_x.to('µm')
        pitch_y = pitch_y.to('µm')

        n_considered_spots_x = int((self.std * self._beam_cutoff / pitch_x).to_reduced_units().m) + 1
        n_considered_spots_y = int((self.std * self._beam_cutoff / pitch_y).to_reduced_units().m) + 1

        x_grid = np.arange(-n_considered_spots_x, n_considered_spots_x+1) * pitch_x.m
        y_grid = np.arange(-n_considered_spots_y, n_considered_spots_y+1) * pitch_y.m

        points = np.dstack(np.meshgrid(x_grid, y_grid)).reshape(-1, 2)
        points = points[points[:, 0]**2 + points[:, 1]**2 < (self.std * self._beam_cutoff).to('µm').m**2]

        fluxes, flux_unit = self.flux_at((0, 0), points, U_('µm'))

        return np.sum(fluxes) * flux_unit


class GaussBeam(IonBeam):
    """Gauss function model of an ion beam."""

    def __init__(self, fwhm: QuantityType, current: QuantityType, beam_cutoff: int = 10):
        """

        Args:
            fwhm (QuantityType): full width half maximum of the beam profile.
            current (QuantityType): total beam current
            beam_cutoff (int, optional): beam cut-off, default to 10.
        """
        super().__init__(beam_cutoff)

        self._fwhm = fwhm
        self._current = current

        self._std = fwhm.to('µm') / (2 * np.sqrt(2 * np.log(2)))

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}'
            f'(fwhm={self._fwhm!r}, current={self._current!r}, beam_cutoff={self._beam_cutoff})'
        )

    @property
    def std(self) -> QuantityType:
        return self._std

    def flux_at(
        self,
        position: VectorLike, beam_maxima: Union[List[VectorLike], VectorLike],
        position_unit: UnitType
    ) -> Union[Tuple[np.ndarray, UnitType], QuantityType]:
        # pylint: disable=invalid-name
        try:
            beam_maxima = Vector(beam_maxima)  # type: ignore
        except VectorValueError:
            try:
                beam_maxima = np.asarray([Vector(vec) for vec in beam_maxima])
            except VectorValueError:
                raise ValueError('I do not understand beam_maxima\' type.')  # pylint: disable=raise-missing-from

        position = Vector(position) * scale_factor(U_('µm'), position_unit)

        beam_maxima *= scale_factor(U_('µm'), position_unit)

        # beam_maxima = np.asarray(beam_maxima)

        if isinstance(beam_maxima, Vector):
            x = position.x - beam_maxima[0]
            y = position.y - beam_maxima[1]
        else:
            x = position.x - beam_maxima[:, 0]  # type: ignore
            y = position.y - beam_maxima[:, 1]  # type: ignore

        two_sigma_squared = 2 * (self._std * self._std).m

        # sqrt(2 * np.pi) ?
        amplitude = self._current.to('pA') / (2 * np.pi * self._std * self._std)

        fluxes = np.exp(-(x * x + y * y) / two_sigma_squared)

        if isinstance(beam_maxima, Vector):
            return fluxes * amplitude

        return fluxes * amplitude.m, amplitude.units
