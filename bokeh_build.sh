#! /bin/sh

bokeh build --rebuild bokeh-measuretool

cp bokeh-measuretool/dist/bokeh-measuretool.min.js src/fibomat/default_backends