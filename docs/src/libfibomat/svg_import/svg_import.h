
#ifndef LIBFIBOMAT_SVG_IMPORT_H
#define LIBFIBOMAT_SVG_IMPORT_H

#include "common.h"

#include <string>

#include <pybind11/pybind11.h>
#include <pybind11_json/pybind11_json.hpp>

namespace py = pybind11;

#include "nlohmann/json.hpp"
using json = nlohmann::json;

#include <rapidxml_ns/rapidxml_ns.hpp>
#include <rapidxml_ns/rapidxml_ns_utils.hpp>
#include <svgpp/policy/xml/rapidxml_ns.hpp>


#include "base_context.h"
#include "document.h"

namespace
{
    py::object load_svg(char* data)
    {
        rapidxml_ns::xml_document<> doc;
        doc.parse<0>(data);

        if (rapidxml_ns::xml_node<> * xml_root_element = doc.first_node("svg"))
        {
            json svg;

            svgimport::base_context context(svg);
            svgimport::document_traversal_t::load_document(xml_root_element, context);

            return svg;
        }

        throw std::runtime_error("Could not load svg file.");
    }
}

namespace fibomat
{
    py::object load_svg_from_file(std::string filename)
    {
        rapidxml_ns::file<> file(filename.c_str());
        return load_svg(file.data());
    }

    py::object load_svg_from_str(std::string svg)
    {
        if (!svg.empty())
        {
            return load_svg(&svg[0]);
        }

        throw std::runtime_error("svg string is empty");
    }

}

#endif //LIBFIBOMAT_SVG_IMPORT_H
