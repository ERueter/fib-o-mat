//
// Created by yorgo on 05/06/20.
//

#ifndef SVGIMPORT_TRAITS_H
#define SVGIMPORT_TRAITS_H

#include <boost/mpl/set.hpp>

#include "svgpp/svgpp.hpp"

namespace svgimport
{
    typedef boost::mpl::set4<
        svgpp::tag::element::circle,
        svgpp::tag::element::ellipse,
        svgpp::tag::element::line,
        svgpp::tag::element::rect
    > basic_shape_elements;

    typedef boost::mpl::set3<
        // svgpp::tag::element::rect,
        svgpp::tag::element::path,
        svgpp::tag::element::polyline,
        svgpp::tag::element::polygon
    > path_elements;
}

#endif //SVGIMPORT_TRAITS_H
