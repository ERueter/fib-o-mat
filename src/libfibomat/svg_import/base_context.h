//
// Created by yorgo on 02/06/20.
//

#ifndef SVGIMPORT_BASE_CONTEXT_H
#define SVGIMPORT_BASE_CONTEXT_H

#include <vector>
#include <string>
#include <map>

#include <boost/array.hpp>

#include "nlohmann/json.hpp"
using json = nlohmann::json;

#include <svgpp/svgpp.hpp>

#include "transformable.h"

namespace svgimport
{
    typedef std::pair<std::string, std::map<std::string, double>> element_t;

    typedef std::map<std::string, double> attribute_t;

    typedef std::vector<std::tuple<std::string, transform_t, attribute_t>> element_vector_t;


    class base_context : public transformable
    {
    public:
        base_context(json& svg)
            : m_svg(svg)
        {
            m_svg = json::array();
            // set dpi to 90 for now
            m_length_factory.set_absolute_units_coefficient(90., svgpp::tag::length_units::in());
        }

        base_context(base_context& parent)
            : m_svg(parent.m_svg), transformable(parent), m_length_factory(parent.m_length_factory)
        {

        }

        template <typename T>
        void set(svgpp::tag::attribute::id, T id_iter)
        {
            // throw std::runtime_error("void set(svgpp::tag::attribute::id, T id)");
            // m_id = std::string(std::begin(id_iter), std::end(id_iter));
            m_current_element["id"] = std::string(std::begin(id_iter), std::end(id_iter));
        }

        void set_viewport(double viewport_x, double viewport_y, double viewport_width, double viewport_height)
        {
            m_length_factory.set_viewport_size(viewport_width, viewport_height);
        }

        void set_viewbox_size(double viewbox_width, double viewbox_height)
        {
            m_length_factory.set_viewport_size(viewbox_width, viewbox_height);
        }

        void disable_rendering()
        {
            throw std::runtime_error("disable_rendering()");
        }

        void on_enter_element(svgpp::tag::element::any) const {}
        void on_exit_element()
        {
            m_current_element["transformations"] = m_transformations;

            m_svg.push_back(m_current_element);
//            m_elements.push_back(std::make_tuple(
//                m_current_element.first,
//                m_transform,
//                m_current_element.second
//            ));
        }

        typedef svgpp::factory::length::unitless<> length_factory_type;

        length_factory_type const & length_factory() const
        {
            return m_length_factory;
        }

    protected:
        //element_vector_t& m_elements;
        json& m_svg;

        json m_current_element;
    private:
        length_factory_type m_length_factory;
    };
}


#endif //SVGIMPORT_BASE_CONTEXT_H
