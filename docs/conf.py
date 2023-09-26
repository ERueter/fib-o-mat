# -*- coding: utf-8 -*-

import sys
import os
import shutil
import inspect
from os.path import basename, dirname, join
from uuid import uuid4


from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.parsers.rst.directives import unchanged

from sphinx.errors import SphinxError
from sphinx.util import logging

from sphinx.ext.napoleon import GoogleDocstring

from bokeh.application.handlers.code_runner import CodeRunner
from bokeh.sphinxext.example_handler import ExampleHandler
from bokeh.model import Model
from bokeh.document import Document
from bokeh.embed import autoload_static
from bokeh.sphinxext.util import get_sphinx_resources


# monkey patch napoleon internal to render custom tags properly
# https://stackoverflow.com/questions/17478040/how-can-i-generate-documentation-for-a-python-property-setter-using-sphinx
def mp_parse_generic_section(self, section: str, use_admonition: bool):  # -> List[str]:
    lines = self._strip_empty(self._consume_to_next_section())
    lines = self._dedent(lines)

    if use_admonition:
        header = '.. admonition:: %s' % section
        lines = self._indent(lines, 3)
    else:
        # header = '.. admonition:: %s' % section
        header = f':{section}:'
        lines = self._indent(lines, 3)

    if lines:
        return [header, ''] + lines + ['']
    else:
        return [header, '']

GoogleDocstring._parse_generic_section = mp_parse_generic_section


# monkey patch bokeh-plots to allow to pass an hint in argv that a script is during sphinx build
def new_init(self, source, filename):
    super(ExampleHandler, self).__init__(self)
    self._runner = CodeRunner(source, filename, ['sphinx-build'])

ExampleHandler.__init__ = new_init


# customize bokeh-plot command to include link to source.
class CustomBokehPlotDirective(Directive):
    has_content = True
    optional_arguments = 1

    option_spec = {
        "url": unchanged
    }

    def run(self):
        from sphinx.locale import _
        # from bokeh.sphinxext.bokeh_plot import _process_script

        log = logging.getLogger(__name__)

        def _process_script(source, filename, env, js_name, use_relative_paths=False):
            # Explicitly make sure old extensions are not included until a better
            # automatic mechanism is available
            Model._clear_extensions()

            run_source = source

            c = ExampleHandler(source=run_source, filename=filename)
            d = Document()

            # We may need to instantiate deprecated objects as part of documenting
            # them in the reference guide. Suppress any warnings here to keep the
            # docs build clean just for this case
            # with warnings.catch_warnings():
            #     if "reference" in env.docname:
            #         warnings.filterwarnings("ignore", category=BokehDeprecationWarning)
            c.modify_document(d)

            if c.error:
                raise RuntimeError(c.error_detail)

            resources = get_sphinx_resources()
            js_path = join(env.bokeh_plot_auxdir, js_name)
            js, script = autoload_static(d.roots[0], resources, js_name)

            with open(js_path, "w") as f:
                f.write(js)

            return (script, js, js_path, source)

        env = self.state.document.settings.env

        # filename *or* python code content, but not both
        if self.arguments and self.content:
            raise SphinxError("bokeh-plot:: directive can't have both args and content")

        # need docname not to look like a path
        docname = env.docname.replace("/", "-")

        if self.content:
            log.debug(f"[bokeh-plot] handling inline example in {env.docname!r}")
            path = env.bokeh_plot_auxdir  # code runner just needs any real path
            source = "\n".join(self.content)
        else:
            try:
                log.debug(f"[bokeh-plot] handling external example in {env.docname!r}: {self.arguments[0]}")
                path = self.arguments[0]
                if not path.startswith("/"):
                    path = join(env.app.srcdir, path)
                source = open(path).read()
            except Exception as e:
                raise SphinxError(f"{env.docname}: {e!r}")

        js_name = f"bokeh-plot-{uuid4().hex}-external-{docname}.js"

        try:
            (script, js, js_path, source) = _process_script(source, path, env, js_name)
        except Exception as e:
            raise RuntimeError(f"Sphinx bokeh-plot exception: \n\n{e}\n\n Failed on:\n\n {source}") from e
        env.bokeh_plot_files[js_name] = (script, js, js_path, source, dirname(env.docname))

        # use the source file name to construct a friendly target_id
        target_id = f"{env.docname}.{basename(js_path)}"
        target = nodes.target("", "", ids=[target_id])
        result = [target]

        result += [nodes.raw("", script, format="html")]

        url = self.options.get("url", False)

        if url:
            para = nodes.paragraph()
            para += nodes.Text('Source: ')
            para += nodes.reference('', _(url), internal=False, refuri=url, reftitle=_('source code'))
            result += [para]

        return result


# build reference

__location__ = os.path.join(os.getcwd(), os.path.dirname(
    inspect.getfile(inspect.currentframe())))


try:  # for Sphinx >= 1.7
    from sphinx.ext import apidoc
except ImportError:
    from sphinx import apidoc

output_dir = os.path.join(__location__, "reference")
module_dir = os.path.join(__location__, "../fibomat")
try:
    shutil.rmtree(output_dir)
except FileNotFoundError:
    pass

try:
    import sphinx
    from pkg_resources import parse_version

    cmd_line = f"sphinx-apidoc -M -e -f -o {output_dir} {module_dir}"
    # cmd_line = cmd_line_template.format(outputdir=output_dir, moduledir=module_dir)

    args = cmd_line.split(" ")
    if parse_version(sphinx.__version__) >= parse_version('1.7'):
        args = args[1:]

    apidoc.main(args)
except Exception as e:
    print("Running `sphinx-apidoc` failed!\n{}".format(e))



import fibomat

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = '3.0'

extensions = [
    'sphinx.ext.todo', 'sphinx.ext.viewcode', 'sphinx.ext.coverage', 'sphinx.ext.mathjax',
    'sphinxemoji.sphinxemoji', 'sphinx.ext.intersphinx',
    'bokeh.sphinxext.bokeh_plot',   'sphinx.ext.autodoc', 'sphinx.ext.autosectionlabel'
]

# autodoc config
autodoc_default_options = {
    'special-members': '__init__',
    'member-order': 'bysource',
    'undoc-members': True,
    'members': True,
    # 'exclude-members': '__weakref__,__dict__,__module__,__dataclass_fields__,__annotations__,__dataclass_params__,__orig_bases__,__parameters__',
    'show-inheritance': True,
    'inherited-members': True,
    'ignore-module-all': True
}

# autosectionlabel config
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 4

# todo_package config
todo_include_todos = True

# napoleon config
# napoleon_use_param = False
napoleon_custom_sections = ['access']
# napoleon_custom_sections = ['access']

# sphinxemoji
sphinxemoji_style = 'twemoji'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'fibomat'
copyright = u'2020 fib-o-mat Contributors'

exclude_patterns = ['_build']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Options for HTML output ---------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'pydata_sphinx_theme'

try:
    from fibomat import __version__ as version
except ImportError:
    pass
else:
    release = version

html_logo = "logo/fibomat.png"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
# html_favicon = None

html_static_path = ['_static']

# If true, links to the reST sources are added to the pages.
# html_show_sourcelink = True

# html_context = {
#     "doc_path": "https://gitlab.com/viggge/fib-o-mat/-/tree/master/docs"
# }

html_theme_options = {
    # "use_edit_page_button": True,
    "external_links": [
        {"name": "Gitlab", "url": "https://gitlab.com/viggge/fib-o-mat"},
    ],
    'show_toc_level': 4
}

# This is the file name suffix for HTML files (e.g. ".xhtml").
# html_file_suffix = None

# Output file base name for HTML help builder.
htmlhelp_basename = 'fibomat-doc'

# -- External mapping ------------------------------------------------------------
python_version = '.'.join(map(str, sys.version_info[0:2]))
intersphinx_mapping = {
    'sphinx': ('http://www.sphinx-doc.org/en/stable', None),
    'python': ('https://docs.python.org/' + python_version, None),
    'matplotlib': ('https://matplotlib.org', None),
    'numpy': ('https://docs.scipy.org/doc/numpy', None),
    'sklearn': ('http://scikit-learn.org/stable', None),
    'pandas': ('http://pandas.pydata.org/pandas-docs/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'sympy': ('https://docs.sympy.org/dev', None),
}


def setup(app):
    app.add_css_file('todo.css')
    app.add_directive('bokeh-plot-link', CustomBokehPlotDirective)
