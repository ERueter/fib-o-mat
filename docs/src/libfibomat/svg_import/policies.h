//
// Created by yorgo on 09/06/20.
//

#ifndef SVGIMPORT_POLICIES_H
#define SVGIMPORT_POLICIES_H

#include "svgpp/svgpp.hpp"

namespace svgimport
{
    struct basic_shapes_policy : svgpp::policy::basic_shapes::raw
    {
        typedef boost::mpl::set<
        svgpp::tag::element::ellipse,
        svgpp::tag::element::rect,
        svgpp::tag::element::line,
        svgpp::tag::element::circle> collect_attributes;

        typedef boost::mpl::set<
        // svgpp::tag::element::rect,
        svgpp::tag::element::polyline,
        svgpp::tag::element::polygon
        > convert_to_path;

        // static const bool convert_only_rounded_rect_to_path = true;
    };

    struct path_policy : svgpp::policy::path::raw
    {
        static const bool absolute_coordinates_only = true;
        static const bool no_ortho_line_to = true;
        static const bool no_quadratic_bezier_shorthand = true;
        static const bool no_cubic_bezier_shorthand = true;
        static const bool quadratic_bezier_as_cubic = true;
        static const bool arc_as_cubic_bezier = false;
    };

    struct attribute_traversal_policy : svgpp::policy::attribute_traversal::default_policy
    {
        typedef boost::mpl::if_<
        // If element is 'svg' or 'symbol'...
        boost::mpl::has_key<
            boost::mpl::set<
            svgpp::tag::element::svg,
            svgpp::tag::element::symbol
        >,
        boost::mpl::_1
        >,
        boost::mpl::vector<
            // ... load viewport-related attributes first ...
            svgpp::tag::attribute::x,
            svgpp::tag::attribute::y,
            svgpp::tag::attribute::width,
            svgpp::tag::attribute::height,
            svgpp::tag::attribute::viewBox,
            svgpp::tag::attribute::preserveAspectRatio,
            // ... notify library, that all viewport attributes that are present was loaded.
            // It will result in call to BaseContext::set_viewport and BaseContext::set_viewbox_size
            svgpp::notify_context<svgpp::tag::event::after_viewport_attributes>
        >::type,
        boost::mpl::empty_sequence
        > get_priority_attributes_by_element;
    };
}

#endif //SVGIMPORT_POLICIES_H
