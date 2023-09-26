//
// Created by yorgo on 09/06/20.
//

#ifndef SVGIMPORT_DOCUMENT_H
#define SVGIMPORT_DOCUMENT_H

#include "svgpp/svgpp.hpp"

#include "transformable.h"
#include "base_context.h"
#include "child_contexts.h"
#include "basic_shape_context.h"
#include "policies.h"

namespace svgimport
{
    typedef
    boost::mpl::set28<
        svgpp::tag::attribute::id,
        // Transform attribute
        svgpp::tag::attribute::transform,
        // Viewport attributes
        svgpp::tag::attribute::x,
        svgpp::tag::attribute::y,
        svgpp::tag::attribute::width,
        svgpp::tag::attribute::height,
        svgpp::tag::attribute::viewBox,
        svgpp::tag::attribute::preserveAspectRatio,
        // Shape attributes for each shape element
        boost::mpl::pair<svgpp::tag::element::path, svgpp::tag::attribute::d>,
        boost::mpl::pair<svgpp::tag::element::rect, svgpp::tag::attribute::x>,
        boost::mpl::pair<svgpp::tag::element::rect, svgpp::tag::attribute::y>,
        boost::mpl::pair<svgpp::tag::element::rect, svgpp::tag::attribute::width>,
        boost::mpl::pair<svgpp::tag::element::rect, svgpp::tag::attribute::height>,
        boost::mpl::pair<svgpp::tag::element::rect, svgpp::tag::attribute::rx>,
        boost::mpl::pair<svgpp::tag::element::rect, svgpp::tag::attribute::ry>,
        boost::mpl::pair<svgpp::tag::element::circle, svgpp::tag::attribute::cx>,
        boost::mpl::pair<svgpp::tag::element::circle, svgpp::tag::attribute::cy>,
        boost::mpl::pair<svgpp::tag::element::circle, svgpp::tag::attribute::r>,
        boost::mpl::pair<svgpp::tag::element::ellipse, svgpp::tag::attribute::cx>,
        boost::mpl::pair<svgpp::tag::element::ellipse, svgpp::tag::attribute::cy>,
        boost::mpl::pair<svgpp::tag::element::ellipse, svgpp::tag::attribute::rx>,
        boost::mpl::pair<svgpp::tag::element::ellipse, svgpp::tag::attribute::ry>,
        boost::mpl::pair<svgpp::tag::element::line, svgpp::tag::attribute::x1>,
        boost::mpl::pair<svgpp::tag::element::line, svgpp::tag::attribute::y1>,
        boost::mpl::pair<svgpp::tag::element::line, svgpp::tag::attribute::x2>,
        boost::mpl::pair<svgpp::tag::element::line, svgpp::tag::attribute::y2>,
        boost::mpl::pair<svgpp::tag::element::polyline, svgpp::tag::attribute::points>,
        boost::mpl::pair<svgpp::tag::element::polygon, svgpp::tag::attribute::points>
    >::type processed_attributes_t;

    typedef
    boost::mpl::set<
        svgpp::tag::element::svg,
        svgpp::tag::element::g,
        svgpp::tag::element::ellipse,
        svgpp::tag::element::rect,
        svgpp::tag::element::line,
        svgpp::tag::element::circle,
        svgpp::tag::element::path,
        svgpp::tag::element::polyline,
        svgpp::tag::element::polygon
    >::type processed_elements_t;

    typedef svgpp::document_traversal<
        svgpp::processed_elements<svgimport::processed_elements_t>,
        svgpp::processed_attributes<svgimport::processed_attributes_t>,
        svgpp::basic_shapes_policy<svgimport::basic_shapes_policy>,
        svgpp::context_factories<svgimport::child_context_factories>,
        svgpp::transform_policy<svgpp::policy::transform::minimal>,
        svgpp::error_policy<svgpp::policy::error::raise_exception<svgimport::base_context>>,
        svgpp::viewport_policy<svgpp::policy::viewport::as_transform>,
        svgpp::length_policy<svgpp::policy::length::forward_to_method<svgimport::base_context>>,
        svgpp::attribute_traversal_policy<svgimport::attribute_traversal_policy>,
        svgpp::path_policy<svgimport::path_policy>
    > document_traversal_t;
}

#endif //SVGIMPORT_DOCUMENT_H
