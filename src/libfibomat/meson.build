


includes = include_directories('.', 'cavc/include', 'pybind11/include')

srcs = [
    'libfibomat.cpp',
]

compiler = meson.get_compiler('cpp')

if host_machine.system() == 'windows' and compiler.get_id() == 'gcc'
    # this statically links libgcc and listdc++ to the pyd file. It is otherwise not loadable
    # without the specific libgcc dll and libstdc++ files installed on the target machine.
    # https://github.com/pybind/pybind11/issues/2010#issuecomment-1288514493
    # https://github.com/mesonbuild/meson/issues/10775
    # link_args = ['-static-libgcc', '-static-libstdc++']
    add_project_link_arguments('-static', language: 'cpp')
endif

py.extension_module(
    '_libfibomat',
    srcs,
    install: true,
    subdir: 'fibomat',
    include_directories: includes,
)