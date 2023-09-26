import difflib
from typing import Type, Dict

from fibomat.backend.backendbase import BackendBase


class Registry:
    """Class is used to register and store backends for exporting.

    Usually, you do not have to create an instance yourself and you rather
    should use the predefined instance :data:`registry`.
    """
    def __init__(self):
        self._backends: Dict[str, Type[BackendBase]] = {}

    @property
    def backends(self) -> Dict[str, Type[BackendBase]]:
        """Dict[str, Type[BackendBase]]: registered backends"""
        return self._backends

    def register(self, class_: Type[BackendBase], name: str) -> Type[BackendBase]:
        """Register the the backend class `class_` with name `name`.
        `name` has to be used in :meth:`get` to retrieve the backend
        class.

        .. note:: `name` can by arbitrary and any other registered
                  backend with the same name gets overwritten.

        Args:
            class_ (typing.Type[BackendBase]): backend class to be registered
            name (str): name of the backend

        Raises:
            TypeError: Raised if `class_` is not a subclass of
                       :class:`~fibomat.backend.backendbase.BackendBase`.
        """
        if not issubclass(class_, BackendBase):
            raise TypeError("class_ must be a subclass of BackendBase")

        self._backends[name] = class_

        return class_

    def get(self, name: str) -> Type[BackendBase]:
        """Returns the registered backend with name `name`

        Args:
            name (str): Name of backend

        Returns:
            typing.Type[BackendBase]: Backend class
        """

        if name in self._backends:
            return self._backends[name]
        else:
            matches = difflib.get_close_matches(name, self._backends.keys())
            error_massage = f'No backend found with name "{name}".'
            if matches:
                error_massage += f' Did you mean "{matches[0]}"?'
            raise KeyError(error_massage)


registry = Registry()
""":class:`Registry` instance"""

# def register_backend(name: str) -> typing.Callable:
#     """Use this function as a decorator to register a new backend for exporting::
#
#         @register_backend("mycoolexporter")
#         class MyCoolExporter(BaseBackend):
#             ...
#
#
#     Note that any other registered backend with the same name gets overwritten.
#
#     Args:
#         name (str): Name of the backend.
#     """
#     print('registering', name)
#     return functools.partial(registry.register, name=name)
