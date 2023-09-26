


includes = include_directories('.', 'cavc/include', 'pybind11/include')

srcs = [
    'libfibomat.cpp',
]

py.extension_module(
    '_libfibomat',
    srcs,
    install: true,
    subdir: 'fibomat',
    include_directories: includes,
)