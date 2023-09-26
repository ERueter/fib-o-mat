#ifndef FIB_O_MAT_CURVE_TOOLS_H
#define FIB_O_MAT_CURVE_TOOLS_H

#include "cavc/polyline.hpp"
#include "cavc/polylineoffset.hpp"
#include "cavc/polylineintersects.hpp"
#include "cavc/polylinecombine.hpp"
#include "cavc/polylineoffset.hpp"
#include "cavc/polylineoffsetislands.hpp"


namespace fibomat
{
    [[nodiscard]] std::vector<cavc::PlineIntersect<double>> raw_self_intersections(const cavc::Polyline<double>& poly) {
        cavc::StaticSpatialIndex<double> spatial_index = cavc::createApproxSpatialIndex(poly);
        std::vector<cavc::PlineIntersect<double>> intersections;
        cavc::allSelfIntersects(poly, intersections, spatial_index);
        return intersections;
    }

    [[nodiscard]] cavc::PlineIntersectsResult<double>
    raw_curve_intersections(const cavc::Polyline<double>& poly1, const cavc::Polyline<double>& poly2) {
        cavc::StaticSpatialIndex<double> spatial_index_poly1 = cavc::createApproxSpatialIndex(poly1);
        cavc::PlineIntersectsResult<double> intersections;
        findIntersects(poly1, poly2, spatial_index_poly1, intersections);


//        auto poly1_start = poly1.vertexes()[0].pos();
//        auto poly2_start = poly2.vertexes()[0].pos();
//
//        auto poly1_end = poly1.lastVertex().pos();
//        auto poly2_end = poly2.lastVertex().pos();
//
//        bool poly1_is_closed = poly1.isClosed();
//        bool poly2_is_closed = poly2.isClosed();
//
//        if (cavc::fuzzyEqual(poly1_start, poly2_start) && !poly1_is_closed && !poly2_is_closed) {
//            auto is_coinc = std::find_if(
//                    std::begin(intersections.coincidentIntersects),
//                    std::end(intersections.coincidentIntersects),
//                    [](const cavc::PlineCoincidentIntersect<double> &coinc) {
//                        return coinc.sIndex1 == 0 && coinc.sIndex2 == 0;
//                    }
//            );
//
//            if (is_coinc == std::end(intersections.coincidentIntersects)) {
//                intersections.intersects.push_back({
//                                                           0, 0,
//                                                           poly1_start
//                                                   });
//            }
//        } else {
//            // start-middle
//            if (!poly1_is_closed) {
//                cavc::ClosestPoint<double> closest_point_on_poly2(poly2, poly1_start);
//                if (cavc::utils::fuzzyEqual(closest_point_on_poly2.distance(), 0.)) {
//                    // std::cout << "start-middle: ";
//                    if (!poly2_is_closed || (poly2_is_closed && !cavc::fuzzyEqual(poly1_start, poly2_end))) {
//                        // std::cout << "true\n";
//                        intersections.intersects.push_back({
//                                                                   0, closest_point_on_poly2.index(),
//                                                                   poly1_start
//                                                           });
//                    }
//                }
//            }
//
//
//            // middle-start
//            if (!poly2_is_closed) {
//                cavc::ClosestPoint<double> closest_point_on_poly1(poly1, poly2_start);
//                if (cavc::utils::fuzzyEqual(closest_point_on_poly1.distance(), 0.)) {
//                    // std::cout << "middle-start: ";
//                    if (!poly1_is_closed || (poly1_is_closed && !cavc::fuzzyEqual(poly2_start, poly1_end))) {
//                        // std::cout << "true\n";
//                        intersections.intersects.push_back({
//                                                                   closest_point_on_poly1.index(), 0,
//                                                                   poly2_start
//                                                           });
//                    }
//                }
//            }
//        }

        return intersections;
    }

    [[nodiscard]] cavc::CombineResult<double>
    raw_combine_curves(const cavc::Polyline<double>& poly1, const cavc::Polyline<double>& poly2,
                   cavc::PlineCombineMode mode) {
        return cavc::combinePolylines(poly1, poly2, mode);
    }

    [[nodiscard]] std::vector<cavc::Polyline<double>>
    raw_offset_curve(const cavc::Polyline<double> &input_curve, double delta) {
        return cavc::parallelOffset(input_curve, delta);
    }


    [[nodiscard]] std::pair<std::vector<cavc::Polyline<double>>, std::vector<cavc::Polyline<double>>>
    raw_offset_with_islands(
            const std::vector<cavc::Polyline<double>>& islands,
            const std::vector<cavc::Polyline<double>>& outer_curve,
            double delta
    ) {
        cavc::OffsetLoopSet<double> loop_set; //{{}, std::move(m_input)};
        for (auto &curve : islands) {
            loop_set.cwLoops.push_back({0, curve, cavc::createApproxSpatialIndex(curve)});
        }

        for (auto &curve : outer_curve) {
            loop_set.ccwLoops.push_back({0, curve, cavc::createApproxSpatialIndex(curve)});
        }

        auto res_loop_set = cavc::ParallelOffsetIslands<double>().compute(loop_set, delta);

        std::cout << "offset multi curve\n";
        std::cout << "res_loop_set.cwLoops " << res_loop_set.cwLoops.size() << "\n";
        std::cout << "res_loop_set.ccwLoops " << res_loop_set.ccwLoops.size() << "\n";

        std::vector<cavc::Polyline<double>> res_islands(res_loop_set.cwLoops.size());
        std::transform(
                std::begin(res_loop_set.cwLoops),
                std::end(res_loop_set.cwLoops),
                std::begin(res_islands),
                [](const auto &c) {
                    return std::move(c.polyline);
                }
        );

        std::vector<cavc::Polyline<double>> res_outer(res_loop_set.ccwLoops.size());
        std::transform(
                std::begin(res_loop_set.ccwLoops),
                std::end(res_loop_set.ccwLoops),
                std::begin(res_outer),
                [](const auto &c) {
                    return std::move(c.polyline);
                }
        );

        return {res_islands, res_outer};
    }
}


#endif
