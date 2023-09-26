#include "common.h"

#include <vector>
#include <map>

#include <svgpp/document_traversal.hpp>
#include <rapidxml_ns/rapidxml_ns.hpp>
#include <rapidxml_ns/rapidxml_ns_utils.hpp>
#include <svgpp/policy/xml/rapidxml_ns.hpp>

#include "nlohmann/json.hpp"
using json = nlohmann::json;

#include "document.h"

#define TEXT(x) #x


void loadSvg(rapidxml_ns::xml_node<char> const * xml_root_element)
{
    // svgimport::element_vector_t elements;
    json svg;
    svgimport::base_context context(svg);
    svgimport::document_traversal_t::load_document(xml_root_element, context);

    std::cout << svg.dump(4) << "\n";

//    for (const auto& elem : elements)
//    {
//        std::cout << std::get<0>(elem) << "\n";
//        std::cout << "\ttransformation\n";
//        for (const auto& transform : std::get<1>(elem))
//        {
//            std::cout << "\t\t" << transform.first<< "\n";
//            for (const auto& prop : transform.second)
//            {
//                std::cout << "\t\t\t" << prop.first << ": " << prop.second << "\n";
//            }
//        }
//
//        for (const auto& attr : std::get<2>(elem))
//        {
//            std::cout << "\t" << attr.first << ": " << attr.second << "\n";
//        }
//    }
}


char xml1[] =
    TEXT(<svg xmlns = "http://www.w3.org/2000/svg">)
    TEXT(<rect x=".01" y=".5" width="4.98" height="2.98"/>)
    TEXT(<rect x="100" y="150" width="400" height="200" rx="0" ry="0"/>)
    TEXT(<path style="fill:none;stroke-width:1;stroke-linecap:square;stroke-linejoin:miter;stroke:rgb(0%,0%,0%);stroke-opacity:1;stroke-miterlimit:3.25;" d="M 77.351563 442.894531 L 75.628906 442.894531 " transform="matrix(1,0,0,1,-74,-13)"/>)
    // TEXT(<rect x="100" y="150" width="400" height="200" rx="50"/>)
    // TEXT(<rect x="100" y="150" width="400" height="200" ry="90"/>)
    // TEXT(<rect x="100" y="150" width="400" height="200" rx="70" ry="80"/>)
    TEXT(<ellipse cx="2.5" cy="1.5" rx="2" ry="1"/>)
    TEXT(<line x1="100" y1="300" x2="350" y2="150"/>)
    TEXT(<circle cx="600" cy="200" r="100"/>)
    TEXT(</svg>)
;
const char text[] =
    TEXT(<svg xmlns="http://www.w3.org/2000/svg">)
    TEXT( <rect x="100" y="150" width="400" height="200"/>)
    TEXT(</svg>);

int main()
{
    rapidxml_ns::file<> file("../pattern.svg");


    rapidxml_ns::xml_document<> doc;    // character type defaults to char
    try
    {
        doc.parse<0>(file.data());
        if (rapidxml_ns::xml_node<> * svg_element = doc.first_node("svg"))
        {
            loadSvg(svg_element);
        }
    }
    catch (std::exception const & e)
    {
        std::cerr << "Error loading SVG: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
