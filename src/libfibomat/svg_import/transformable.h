//
// Created by yorgo on 07/06/20.
//

#ifndef SVGIMPORT_TRANSFORMABLE_H
#define SVGIMPORT_TRANSFORMABLE_H

#include <vector>
#include <string>
#include <map>

#include <boost/array.hpp>

#include "nlohmann/json.hpp"
using json = nlohmann::json;



namespace svgimport
{

    typedef std::vector<std::pair<std::string, std::map<std::string, double>>> transform_t;

    class transformable
    {
    public:

        transformable() : m_transformations(json::array())
        {};

        transformable(const transformable &other) : m_transformations(other.m_transformations)
        {}

        template<class Number>
        void transform_matrix(const boost::array<Number, 6> &matrix)
        {
//            m_transform.push_back({
//                "matrix",
//                {{"a", matrix[0]}, {"b", matrix[1]}, {"c", matrix[2]}, {"d", matrix[3]},
//                 {"e", matrix[4]}, {"f", matrix[5]}}
//            });

            m_transformations.push_back({{"type", "matrix"}, {"matrix", matrix}});

        }

        template<class Number>
        void transform_translate(Number tx, Number ty)
        {
//            m_transform.push_back({"translate", {{"tx", ty}, {"ty", ty}}});
            m_transformations.push_back({{"type", "translation"}, {"tx", ty}, {"ty", ty}});
        }

        template<class Number>
        void transform_scale(Number sx, Number sy)
        {
//            m_transform.push_back({"scale", {{"sx", sx}, {"sy", sy}}});
            m_transformations.push_back({{"type", "scale"}, {"sx", sx}, {"sy", sy}});
        }

        template<class Number>
        void transform_rotate(Number angle)
        {
            m_transformations.push_back({{"type", "rotate"}, {"angle", angle}});
//            m_transform.push_back({"rotate", {{"angle", angle}}});
        }

        template<class Number>
        void transform_skew_x(Number angle)
        {
            m_transformations.push_back({{"type", "skew_x"}, {"angle", angle}});
//            m_transform.push_back({"skew_x", {{"angle", angle}}});
        }

        template<class Number>
        void transform_skew_y(Number angle)
        {
            m_transformations.push_back({{"type", "skew_y"}, {"angle", angle}});

//            m_transform.push_back({"skew_y", {{"angle", angle}}});
        }

    protected:
        json m_transformations;
    };

}

#endif //SVGIMPORT_TRANSFORMABLE_H
