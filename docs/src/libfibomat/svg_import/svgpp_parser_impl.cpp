//
// Created by yorgo on 07/06/20.
//
#include "common.h"

#include <svgpp/parser/external_function/parse_all_impl.hpp>
#include <svgpp/factory/unitless_length.hpp>

//
SVGPP_PARSE_PATH_DATA_IMPL(const char *, double)
SVGPP_PARSE_TRANSFORM_IMPL(const char *, double)
// SVGPP_PARSE_PAINT_IMPL(svg_string_t::value_type const *, color_factory_t, svgpp::factory::icc_color::default_factory)
// SVGPP_PARSE_COLOR_IMPL(svg_string_t::value_type const *, color_factory_t, svgpp::factory::icc_color::default_factory)
SVGPP_PARSE_PRESERVE_ASPECT_RATIO_IMPL(const char *)
SVGPP_PARSE_MISC_IMPL(std::string::value_type const *, double)
SVGPP_PARSE_CLIP_IMPL(const char *, svgpp::factory::length::unitless<>)
SVGPP_PARSE_LENGTH_IMPL(const char *, svgpp::factory::length::unitless<>)
