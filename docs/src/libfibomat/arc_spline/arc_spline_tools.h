
#ifndef LIBFIBOMAT_ARC_SPLINE_TOOLS_H
#define LIBFIBOMAT_ARC_SPLINE_TOOLS_H

#include <vector>
#include <utility>
#include <iostream>

#include "arc_spline.h"
#include "raw_tools.h"

namespace fibomat {
    typedef std::vector<std::tuple<std::size_t, std::size_t, std::pair<double, double>>> intersection_t;
    typedef std::vector<std::tuple<std::size_t, std::size_t, std::pair<double, double>, std::pair<double, double>>>
            coincident_t;

    template<typename T>
    arc_spline<T> convert_arcs_to_lines(const arc_spline<T> c, T error)
    {
        return arc_spline<T>(cavc::convertArcsToLines(c.raw_curve(), error));
    }

    template<typename T>
    intersection_t self_intersection(const arc_spline<T> c) {
        intersection_t intersections;

        for (const auto &intersection : raw_self_intersections(c.raw_curve())) {
            intersections.push_back(
                    {intersection.sIndex1, intersection.sIndex2, {intersection.pos.x(), intersection.pos.y()}}
            );
        }

        return intersections;
    }

    template<typename T>
    [[nodiscard]] std::tuple<intersection_t, coincident_t>
    curve_intersections (const arc_spline<T> c1, const arc_spline<T> c2) {

        // TODO: check if curves are empty!!

        auto raw_intersections = raw_curve_intersections(c1.raw_curve(), c2.raw_curve());

        intersection_t intersections;
        for (const auto &intersection : raw_intersections.intersects) {
            intersections.push_back(
                    {intersection.sIndex1, intersection.sIndex2, {intersection.pos.x(), intersection.pos.y()}}
            );
        }


        coincident_t coincidences;
        for (const auto &coincidence : raw_intersections.coincidentIntersects) {
            coincidences.push_back({
                                           coincidence.sIndex1, coincidence.sIndex2,
                                           {coincidence.point1.x(), coincidence.point1.y()},
                                           {coincidence.point2.x(), coincidence.point2.y()}
                                   });
        }
        return {intersections, coincidences};
    }

    template<typename T>
    [[nodiscard]] std::pair<std::vector<arc_spline<T>>, std::vector<arc_spline<T>>>
    combine_curves(const arc_spline<T> c1, const arc_spline<T> c2, std::string mode_name)
    {
        if (!c1.is_closed() || !c2.is_closed())
        {
            throw std::runtime_error("Only closed curves can be combined.");
        }

        cavc::PlineCombineMode mode;

        if (mode_name == "union")
        {
            mode = cavc::PlineCombineMode::Union;
        }
        else if (mode_name == "xor")
        {
            mode = cavc::PlineCombineMode::XOR;
        }
        else if (mode_name == "exclude")
        {
            mode = cavc::PlineCombineMode::Exclude;
        }
        else if (mode_name == "intersect")
        {
            mode = cavc::PlineCombineMode::Intersect;
        }
        else
        {
            throw std::runtime_error("Unknown combining mode.");
        }

        auto combine_result = raw_combine_curves(c1.raw_curve(), c2.raw_curve(), mode);

        std::vector<arc_spline<T>> remaining;
        for (const auto& res_curve : combine_result.remaining)
        {
            remaining.push_back(std::move(res_curve));
        }

        std::vector<arc_spline<T>> subtracted;
        for (const auto& res_curve : combine_result.subtracted)
        {
            subtracted.push_back(std::move(res_curve));
        }

        return {remaining, subtracted};
    }

    template<typename T>
    [[nodiscard]] std::vector<arc_spline<T>>
    offset_curve (const arc_spline<T> input_curve, T delta)
    {
        auto raw_res_curves = raw_offset_curve(input_curve.raw_curve(), delta);

        std::vector<arc_spline<T>> res_curves(raw_res_curves.size());
        std::transform(std::begin(raw_res_curves), std::end(raw_res_curves), std::begin(res_curves), [](const auto& raw){
            // return curve(std::move(c));
            return arc_spline<T>(raw);
        });

        return res_curves;
    }

    template<typename T>
    [[nodiscard]] std::tuple<std::vector<arc_spline<T>>, std::vector<arc_spline<T>>>
    offset_with_islands (std::vector<arc_spline<T>> islands, std::optional<arc_spline<T>> outer_curve, double delta)
    {
        std::vector<cavc::Polyline<double>> raw_islands(islands.size());
        std::transform(std::begin(islands), std::end(islands), std::begin(raw_islands), [](const auto& island){
            // return std::move(c.raw_curve());
            return island.raw_curve();
        });

        std::vector<cavc::Polyline<double>> raw_outer_curve;

        if (outer_curve)
        {
            raw_outer_curve.push_back(outer_curve->raw_curve());
        }

        auto raw_res_curves = raw_offset_with_islands(raw_islands, raw_outer_curve, delta);

        std::vector<arc_spline<T>> res_islands(raw_res_curves.first.size());
        std::transform(std::begin(raw_res_curves.first), std::end(raw_res_curves.first), std::begin(res_islands), [](const auto& raw){
            // return curve(std::move(c));
            return arc_spline<T>(raw);
        });

        std::vector<arc_spline<T>> res_outer_curve(raw_res_curves.second.size());

        if (!raw_res_curves.second.empty())
        {
            // if (raw_res_curves.second.size() != 1)
            // {   
            //     std::cout << raw_res_curves.second.size() << "\n";
            //     throw std::runtime_error("raw_res_curves.second.size() != 1");
            // }

            // res_outer_curve = arc_spline<T>(raw_res_curves.second[0]);
            std::transform(std::begin(raw_res_curves.second), std::end(raw_res_curves.second), std::begin(res_outer_curve), [](const auto& raw){
                // return curve(std::move(c));
                return arc_spline<T>(raw);
            });
        }

        return {res_islands, res_outer_curve};
    }
}

//m.def("trim_curves", [](const arc_spline_t& c1, const arc_spline_t& c2) -> std::pair<std::vector<arc_spline_t>, std::vector<arc_spline_t>>
//    {
//        std::cout << std::boolalpha << c1.is_closed() << "\n";
//
//        if (!c1.is_closed())
//        {
//            std::cout << "now raising" << "\n" << std::flush;
//            throw std::runtime_error("First curve must be closed for trimming.");
//        }
//
//
//
//        // check if start and end of c2 lie on the outside of c1 if c2 is open.
//
//
//
//        auto c2_raw = c2.raw_curve();
//        auto raw_intersections = curve_intersections(c1.raw_curve(), c2_raw);
//
//        if (raw_intersections.intersects.size() == 0)
//        {
//            return {{}, {}};
//        }
//        else
//        {
//            auto intersections = std::move(raw_intersections.intersects);
//
//            // sort intersections by segment indices (sIndex2) of second curve (the one, which will be splitted)
//            std::sort(std::begin(intersections), std::end(intersections), [](const auto& elem1, const auto& elem2)
//            {
//                return elem1.sIndex2 < elem2.sIndex2;
//            });
//            // it could happen that each segment of the second curve has multiple intersections with first curve.
//            // in the following, we sort the points of each segment. because bulge <= 1, we can sort this points
//            // by calculating the Euclidean distance of each intersection point which the start point of the segment
//            // and compare these.
//            auto intersection_iter = std::begin(intersections);
//            auto last_eq_segment_intersection_iter = intersection_iter;
//            // std::size_t last_seg_index = intersection_iter->sIndex2;
//            ++intersection_iter;
//
//            while (intersection_iter != std::end(intersections))
//            {
//                while (intersection_iter->sIndex2 == last_eq_segment_intersection_iter->sIndex2 && intersection_iter != std::end(intersections))
//                {
//                    ++intersection_iter;
//                }
//
//                if (std::distance(last_eq_segment_intersection_iter, intersection_iter) > 1)
//                {
//                    std::cout << "sorting\n";
//                    std::sort(last_eq_segment_intersection_iter, intersection_iter, [&c2_raw](const auto& elem1, const auto& elem2)
//                    {
//                        return cavc::length(elem1.pos - c2_raw[elem1.sIndex2].pos()) < cavc::length(elem2.pos - c2_raw[elem2.sIndex2].pos());
//                    });
//                }
//
//                last_eq_segment_intersection_iter = intersection_iter;
//            }
//
//            auto extract_curve = [](const auto& intersections, const auto& part_curve)
//            {
//                // std::size_t vertex_index = 0;
//                const auto& part_curve_vertices = part_curve.vertexes();
//
//                std::vector<decltype(part_curve_vertices.begin())> find_results;
//
//                for (const auto& intersection : intersections)
//                {
//                    auto find_result = std::find_if(std::begin(part_curve_vertices), std::end(part_curve_vertices), [&intersection](auto vertex)
//                    {
//                        return cavc::fuzzyEqual(intersection.pos, vertex.pos());
//                    });
//
//                    if (find_result != std::end(part_curve_vertices))
//                    {
//                        find_results.push_back(find_result);
//                    }
//                }
//
//                if (find_results.size() != 2)
//                {
//                    throw std::runtime_error("find_results.size() != 2");
//                }
//
//                if (std::distance(find_results[0], find_results[1]) < 0)
//                {
//                    cavc::Polyline<double> extracted;
//
//                    std::size_t i = std::distance(std::begin(part_curve_vertices), find_results[0]);
//
//                    std::size_t mod = part_curve_vertices.size();
//
//                    while (!cavc::fuzzyEqual(part_curve_vertices[i++].pos(), find_results[1]->pos()));
//                    {
//                       extracted.addVertex(part_curve_vertices[i]);
//                    }
//
//                    return arc_spline_t(extracted);
//                }
//                else
//                {
//                     throw std::runtime_error("I don't know how this happened.");
//                }
//            };
//
//
//
//            auto excluded_curves = combine_curves(c1.raw_curve(), c2.raw_curve(), cavc::PlineCombineMode::Exclude).remaining;
//
//            for (const auto& excluded_curve : excluded_curves)
//            {
//                return {{extract_curve(intersections, excluded_curve)}, {}};
//            }
//
//
//
//
//
//            // auto split_curve = [](arc_spline_t to_be_splitted, cavc::Vector2<double> split_point, std::size_t segment) -> std::pair<arc_spline_t, arc_spline_t>
//            // {
//            //     std::cout << "split\n";
//
//            //     std::cout << "to_be_splitted.n_vertices() = " << to_be_splitted.n_vertices() << "\n";
//
//            //     auto to_be_splitted_raw = to_be_splitted.raw_curve().vertexes();
//
//            //     cavc::SplitResult<double> split_res = cavc::splitAtPoint(to_be_splitted_raw.at(segment), to_be_splitted_raw.at(cavc::utils::nextWrappingIndex(segment+1, to_be_splitted_raw)), split_point);
//
//            //     arc_spline_t before_split;
//            //     for (std::size_t i = 0; i < segment+1; ++i)
//            //     {
//            //         std::cout <<  i << " ";
//            //         before_split.add_vertex(to_be_splitted_raw.at(i));
//            //     }
//
//            //     // if (segment > 0)
//            //     // {
//            //     //     std::cout << "segment_1\n";
//            //     before_split.raw_curve().vertexes().at(segment) = split_res.updatedStart;
//            //     //     std::cout << "segment_2\n";
//
//            //     // }
//            //     // else
//            //     // {
//            //     //     before_split.add_vertex(split_res.updatedStart);
//            //     // }
//            //     // before_split.add_vertex(split_res.updatedStart);
//
//            //     before_split.add_vertex(split_res.splitVertex);
//
//            //     arc_spline_t after_split;
//            //     after_split.add_vertex(split_res.splitVertex);
//
//            //     for (std::size_t i = segment+1; i < to_be_splitted_raw.size(); ++i)
//            //     {
//            //         after_split.add_vertex(to_be_splitted_raw.at(i));
//            //     }
//            //     if (to_be_splitted.is_closed())
//            //     {
//            //         after_split.add_vertex(to_be_splitted_raw.at(0));
//            //     }
//
//            //     return {before_split, after_split};
//            // };
//
//
//            // auto split_res = cavc::splitAtPoint(c2_raw[intersections[0].sIndex2], c2_raw[intersections[0].sIndex2+1], intersections[0].pos);
//            // std::cout << intersections[0].sIndex1 << " " << intersections[0].sIndex2 << "; " << intersections[0].pos.x() << ", " << intersections[0].pos.y() << "\n";
//
//            // std::vector<arc_spline_t> parts_of_c2;
//
//            // arc_spline_t residual_curve = c2;
//
//            // int intersection_point_offset = 0;
//
//            // for (const auto& intersection : intersections)
//            // {
//            //     std::cout << intersection.sIndex1 << " " << intersection.sIndex2 << "; " << intersection.pos.x() << ", " << intersection.pos.y() << "\n";
//            //     std::cout << "res_curve.n_vertices() = " << residual_curve.n_vertices() << "\n";
//            //     for (const auto& v : residual_curve.raw_curve().vertexes())
//            //     {
//            //         std::cout << v.pos().x() << " " << v.pos().y() << "\n";
//            //     }
//
//            //     std::cout << "intersection_point_offset = " << intersection_point_offset << "\n";
//
//            //     auto splitted_curve = split_curve(residual_curve, intersection.pos, intersection.sIndex2 + intersection_point_offset);
//            //     parts_of_c2.push_back(splitted_curve.first);
//
//            //     residual_curve = splitted_curve.second;
//
//            //     intersection_point_offset -= (static_cast<long>(splitted_curve.first.n_vertices()) - 2);
//
//            // }
//
//            // parts_of_c2.push_back(residual_curve);
//
//            // std::vector<arc_spline_t> inside;
//            // std::vector<arc_spline_t> outside;
//
//            // // https://stackoverflow.com/a/12075671
//            // bool toggle = false;
//            // std::partition_copy(parts_of_c2.begin(),
//            //                     parts_of_c2.end(),
//            //                     std::back_inserter(outside),
//            //                     std::back_inserter(inside),
//            //                     [&toggle](const auto& ) { return toggle = !toggle; });
//
//
//            // return {inside, outside};
//
//            // arc_spline_t before_split;
//        }
//    });



#endif //LIBFIBOMAT_ARC_SPLINE_TOOLS_H
