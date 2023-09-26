//
// Created by yorgo on 07/06/20.
//

#ifndef SVGIMPORT_PATH_CONTEXT_H
#define SVGIMPORT_PATH_CONTEXT_H

#include <svgpp/svgpp.hpp>

#include "base_context.h"

namespace
{
    bool is_almost_zero(double a)
    {
        const double epsilon = 1e-5;
        return std::abs(a) <= epsilon;
    }
}

namespace svgimport
{
    class path_context : public base_context
    {
    public:
        path_context(base_context& parent) : base_context(parent)
        {

        }

        void path_move_to(double x, double y, svgpp::tag::coordinate::absolute)
        {
            if (!m_first_segment)
            {
                _save_current_segment();
            }
            m_current_pos = {x, y};
            m_current_segment_start = m_current_pos;
        }

        void path_line_to(double x, double y, svgpp::tag::coordinate::absolute)
        {
            m_current_segment.push_back({
                {"type", "line"}, {"start", {m_current_pos.first, m_current_pos.second}, "end", {x, y}}
            });

            m_current_pos = {x, y};
        }

        void path_cubic_bezier_to(
            double x1, double y1,
            double x2, double y2,
            double x, double y,
            svgpp::tag::coordinate::absolute)
        {
            throw std::runtime_error("path_cubic_bezier_to()");

        }

        void path_elliptical_arc_to(
            double rx, double ry, double x_axis_rotation,
            bool large_arc_flag, bool sweep_flag,
            double x, double y,
            svgpp::tag::coordinate::absolute)
        {
            throw std::runtime_error("path_elliptical_arc_to()");
        }

        void path_close_subpath()
        {
            throw std::runtime_error("path_close_subpath()");
//            if (!is_almost_zero(m_current_pos.first - m_current_segment_start.first)
//                || !is_almost_zero(m_current_pos.first - m_current_segment_start.first))
//            {
//                path_line_to(
//                    m_current_segment_start.first, m_current_segment_start.second, svgpp::tag::coordinate::absolute()
//                );
//            }
//
//            _save_current_segment();
        }

        void path_exit()
        {
            m_current_element = {
                {"type", "path"},
            };
        }

    private:

        void _save_current_segment()
        {
            if (!m_first_segment)
            {
                m_first_segment = false;
            }


        }
        // std::vector<> m_path_segments;
        json m_segments{json::array()};
        json m_current_segment{json::array()};
        bool m_first_segment{true};

        std::pair<double, double> m_current_segment_start;
        // move must be first element of path description, so we don't have to initialize it.
        std::pair<double, double> m_current_pos;

    };
}

#endif //SVGIMPORT_PATH_CONTEXT_H
