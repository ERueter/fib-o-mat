project(
    'fib-o-mat',
    'cpp',
    license: 'GPL 3',
    default_options : ['cpp_std=c++17']
)

py = import('python').find_installation(pure: false)


# measuretool = custom_target(
#     'bokeh-measuretool',
#     build_by_default: true,
#     output : 'bokeh-measuretool.min.js',
#     input : [
#         join_paths(meson.current_source_dir(), 'bokeh-measuretool/bokeh.ext.json'),
#         join_paths(meson.current_source_dir(), 'bokeh-measuretool/index.ts'),
#         join_paths(meson.current_source_dir(), 'bokeh-measuretool/measuretool.ts'),
#         join_paths(meson.current_source_dir(), 'bokeh-measuretool/package.json'),
#         join_paths(meson.current_source_dir(), 'bokeh-measuretool/tsconfig.json'),
#     ],
#     install: true,
#     install_dir: py.get_install_dir() / 'fibomat/default_backends',
#     command : [py.path(), join_paths(meson.current_source_dir(), 'bokeh_build.py'), '@PRIVATE_DIR@', '@OUTDIR@', '@OUTPUT@', '@INPUT@'],
# )
# py.install_sources(
#     [join_paths(meson.current_source_dir(), 'src/fibomat/default_backends/bokeh-measuretool.min.js')],
#     subdir: py.get_install_dir() / 'fibomat/default_backends'
# )



subdir('src/libfibomat')
subdir('src/fibomat')
