//
// Created by yorgo on 02/06/20.
//

#ifndef SVGIMPORT_BASIC_SHAPE_CONTEXT_H
#define SVGIMPORT_BASIC_SHAPE_CONTEXT_H

#include "base_context.h"

namespace svgimport
{
    class basic_shape_context : public base_context
    {
    public:
        basic_shape_context(base_context& parent)
            : base_context(parent)
        {

        }

        void set_rect(double x, double y, double width, double height, double rx, double ry)
        {
            m_current_element = {
                {"type", "rect"}, {"x", x}, {"y", y}, {"width", width}, {"height", height}, {"rx", rx}, {"ry", ry}
            };
//            if (!is_almost_zero(rx) || !is_almost_zero(ry))
//            {
//                throw std::runtime_error("!is_almost_zero(rx) || !is_almost_zero(y)");
//            }

//            m_current_element = {
//                "rect",
//                {{"x", x}, {"y", y}, {"width", width}, {"height", height}, {"rx", rx}, {"ry", ry}}
//            };
        };

        void set_line(double x1, double y1, double x2, double y2)
        {
            m_current_element = {
                {"type", "line"}, {"x1", x1}, {"y1", y1}, {"x2", x2}, {"y2", y2}
            };
//            m_current_element = {
//                "line",
//                {{"x1", x1},{"y1", y1}, {"x2", x2},{"y2", y2}}
//            };
        };

        void set_circle(double cx, double cy, double r)
        {
            m_current_element = {
                {"type", "circle"}, {"cx", cx}, {"cy", cy}, {"r", r}
            };
//            m_current_element = {
//                "circle",
//                {{"cx", cx}, {"cy", cy}, {"r", r}}
//            };
        };

        void set_ellipse(double cx, double cy, double rx, double ry)
        {
            m_current_element = {
                {"type", "ellipse"}, {"cx", cx}, {"cy", cy}, {"rx", rx}, {"ry", ry}
            };
//            m_current_element = {
//                "ellipse",
//                {{"cx", cx}, {"cy", cy}, {"rx", rx}, {"ry", ry}}
//            };
        };
    };
}

#endif //SVGIMPORT_BASIC_SHAPE_CONTEXT_H
