#include "cavc/polylineoffset.hpp"

#include "curve_offset/curve_offset.hpp"
#include "curve_tools.h"

template <typename StringT, typename IntersectT>
void print_intersects(StringT type, IntersectT intersects)
{
    std::cout << type << "\n";
    for (auto intersect : intersects.intersects)
    {
        std::cout << intersect.pos.x() << " " << intersect.pos.y() << "\n";
    }
    // std::cout << "\n";
}

cavc::PlineIntersectsResult<double> curve_intersections_with_ends(const cavc::Polyline<double> poly1, const cavc::Polyline<double> poly2)
{

    cavc::StaticSpatialIndex<double> spatial_index_poly1 = cavc::createApproxSpatialIndex(poly1);
    cavc::PlineIntersectsResult<double> intersections;
    findIntersects(poly1, poly2, spatial_index_poly1, intersections);


    // test for star-start, start-end, end-start
    auto poly1_start = poly1.vertexes()[0].pos();
    auto poly2_start = poly2.vertexes()[0].pos();

    auto poly1_end = poly1.lastVertex().pos();
    auto poly2_end = poly2.lastVertex().pos();

    bool poly1_is_closed = poly1.isClosed();
    bool poly2_is_closed = poly2.isClosed();

//    // start-start
//    if (!(poly1_is_closed && poly2_is_closed))
//    {
//        if (cavc::fuzzyEqual(poly1_start, poly2_start))
//        {
//            intersections.intersects.push_back({
//                0, 0,
//                poly1_start
//            });
//        }
//    }
//    // start-end
//    if (!poly1_is_closed)
//    {
//        if (cavc::fuzzyEqual(poly1_start, poly2_end))
//        {
//            intersections.intersects.push_back({
//                0, 0,
//                poly1_start
//            });
//        }
//    }
//    // end-start
//    if (!poly2_is_closed)
//    {
//        if (cavc::fuzzyEqual(poly1_end, poly2_start))
//        {
//            intersections.intersects.push_back({
//                0, 0,
//                poly1_end
//            });
//        }
//     }

     if (cavc::fuzzyEqual(poly1_start, poly2_start) && !poly1_is_closed && !poly2_is_closed)
     {
        auto is_coinc = std::find_if(
            std::begin(intersections.coincidentIntersects),
            std::end(intersections.coincidentIntersects),
            [](const cavc::PlineCoincidentIntersect<double>& coinc)
            {
                return coinc.sIndex1 == 0 && coinc.sIndex2 == 0;
            }
        );

        if (is_coinc == std::end(intersections.coincidentIntersects))
        {
            intersections.intersects.push_back({
                0, 0,
                poly1_start
            });
        }
     }
     else
     {
         // start-middle
         if (!poly1_is_closed)
         {
             cavc::ClosestPoint<double> closest_point_on_poly2(poly2, poly1_start);
             if (cavc::utils::fuzzyEqual(closest_point_on_poly2.distance(), 0.))
             {
                // std::cout << "start-middle: ";
                if (!poly2_is_closed || (poly2_is_closed && !cavc::fuzzyEqual(poly1_start, poly2_end)))
                {
                    // std::cout << "true\n";
                    intersections.intersects.push_back({
                        0, closest_point_on_poly2.index(),
                        poly1_start
                    });
                }
             }
         }


         // middle-start
         if (!poly2_is_closed)
         {
             cavc::ClosestPoint<double> closest_point_on_poly1(poly1, poly2_start);
             if (cavc::utils::fuzzyEqual(closest_point_on_poly1.distance(), 0.))
             {
                // std::cout << "middle-start: ";
                if (!poly1_is_closed || (poly1_is_closed && !cavc::fuzzyEqual(poly2_start, poly1_end)))
                {
                    // std::cout << "true\n";
                    intersections.intersects.push_back({
                        closest_point_on_poly1.index(), 0,
                        poly2_start
                    });
                }
             }
         }
     }

    return intersections;
}


int main(int argc, char *argv[]) {
    {
        std::cout << "1.\n";
        cavc::Polyline<double> l1;
        l1.addVertex(0., 0., 0.);
        l1.addVertex(1., 0., 0.);
        l1.isClosed() = false;

        cavc::Polyline<double> l2;
        l2.addVertex(0., 0., 0.);
        l2.addVertex(0., 1., 0.);
        l1.isClosed() = false;

        print_intersects("cavc: line-line start-start", curve_intersections(l1, l2));
        print_intersects("custom: line-line start-start", curve_intersections_with_ends(l1, l2));
        std::cout << "\n";

    }

    {
        std::cout << "2.\n";
        cavc::Polyline<double> l1;
        l1.addVertex(-1., 0., 0.);
        l1.addVertex(0., 0., 0.);
        l1.isClosed() = false;

        cavc::Polyline<double> l2;
        l2.addVertex(0., 0., 0.);
        l2.addVertex(1., 0., 0.);
        l1.isClosed() = false;

        print_intersects("cavc: line-line end-start", curve_intersections(l1, l2));
        print_intersects("custom: line-line end-start", curve_intersections_with_ends(l1, l2));
        std::cout << "\n";
    }

    {
        std::cout << "3.\n";

        cavc::Polyline<double> l1;
        l1.addVertex(0., 0., 0.);
        l1.addVertex(1., 0., 0.);
        l1.isClosed() = false;

        cavc::Polyline<double> l2;
        l2.addVertex(-1., 0., 0.);
        l2.addVertex(0., 0., 0.);
        l1.isClosed() = false;

        print_intersects("cavc: line-line start-end", curve_intersections(l1, l2));
        print_intersects("custom: line-line start-end", curve_intersections_with_ends(l1, l2));
        std::cout << "\n";
    }

    {
        std::cout << "4.\n";

        cavc::Polyline<double> l1;
        l1.addVertex(-1., 0., 0.);
        l1.addVertex(0., 0., 0.);
        l1.isClosed() = false;

        cavc::Polyline<double> l2;
        l2.addVertex(1., 0., 0.);
        l2.addVertex(0., 0., 0.);
        l1.isClosed() = false;

        print_intersects("cavc: line-line end-end", curve_intersections(l1, l2));
        print_intersects("custom: line-line end-end", curve_intersections_with_ends(l1, l2));
        std::cout << "\n";
    }

    {
        std::cout << "5.\n";

        cavc::Polyline<double> l;
        l.addVertex(1., 0., 0.);
        l.addVertex(0., 0., 0.);
        l.isClosed() = false;

        cavc::Polyline<double> circle;
        circle.addVertex(1., 0., 1.);
        circle.addVertex(-1., 0., 1.);
        circle.isClosed() = true;

        print_intersects("cavc: circle-line start-start", curve_intersections(circle, l));
        print_intersects("custom: circle-line start-start", curve_intersections_with_ends(circle, l));
        std::cout << "\n";
    }

    {
        std::cout << "6.\n";

        cavc::Polyline<double> l;
        l.addVertex(0., 0., 0.);
        l.addVertex(1., 0., 0.);
        l.isClosed() = false;

        cavc::Polyline<double> circle;
        circle.addVertex(1., 0., 1.);
        circle.addVertex(-1., 0., 1.);
        circle.isClosed() = true;

        print_intersects("cavc: circle-line start-end", curve_intersections(circle, l));
        print_intersects("custom: circle-line start-end", curve_intersections_with_ends(circle, l));
        std::cout << "\n";
    }



    {
        std::cout << "7.\n";
        cavc::Polyline<double> l;
        l.addVertex(0., 1., 0.);
        l.addVertex(0., 0., 0.);
        l.isClosed() = false;

        cavc::Polyline<double> circle;
        circle.addVertex(1., 0., 1.);
        circle.addVertex(-1., 0., 1.);
        circle.isClosed() = true;

        print_intersects("cavc: circle-line middle-start", curve_intersections(circle, l));
        print_intersects("custom: circle-line middle-start", curve_intersections_with_ends(circle, l));
        std::cout << "\n";
    }

    {
        std::cout << "8.\n";
        cavc::Polyline<double> l;
        l.addVertex(0., 0., 0.);
        l.addVertex(0., 1., 0.);
        l.isClosed() = false;

        cavc::Polyline<double> circle;
        circle.addVertex(1., 0., 1.);
        circle.addVertex(-1., 0., 1.);
        circle.isClosed() = true;

        print_intersects("cavc: circle-line middle-end", curve_intersections(circle, l));
        print_intersects("custom: circle-line middle-end", curve_intersections_with_ends(circle, l));
        std::cout << "\n";
    }

    {
        std::cout << "9.\n";
        cavc::Polyline<double> circle1;
        circle1.addVertex(1., 0., 1.);
        circle1.addVertex(-1., 0., 1.);
        circle1.isClosed() = true;

        cavc::Polyline<double> circle2;
        circle2.addVertex(1., 0., 1.);
        circle2.addVertex(2., 0., 1.);
        circle2.isClosed() = true;

        print_intersects("cavc: circle-circle start-start", curve_intersections(circle1, circle2));
        print_intersects("custom: circle-circle start-start", curve_intersections_with_ends(circle1, circle2));
        std::cout << "\n";
    }

    {
        std::cout << "10.\n";
        cavc::Polyline<double> circle1;
        circle1.addVertex(1., 0., 1.);
        circle1.addVertex(-1., 0., 1.);
        circle1.isClosed() = true;

        cavc::Polyline<double> circle2;
        circle2.addVertex(2., 0., 1.);
        circle2.addVertex(1., 0., 1.);
        circle2.isClosed() = true;

        print_intersects("cavc: circle-circle start-middle", curve_intersections(circle1, circle2));
        print_intersects("custom: circle-circle start-middle", curve_intersections_with_ends(circle1, circle2));
        std::cout << "\n";
    }

    {
         std::cout << "11.\n";
        cavc::Polyline<double> l;
        l.addVertex(0., 0., 0.);
        l.addVertex(2., 0., 0.);
        l.isClosed() = false;

        cavc::Polyline<double> circle;
        circle.addVertex(1., 0., 1.);
        circle.addVertex(-1., 0., 1.);
        circle.isClosed() = true;

        print_intersects("cavc: circle-line start-middle", curve_intersections(circle, l));
        print_intersects("custom: circle-line start-middle", curve_intersections_with_ends(circle, l));
        std::cout << "\n";
    }
    {
        std::cout << "12.\n";
        cavc::Polyline<double> l;
        l.addVertex(0., 1., 0.);
        l.addVertex(1., 0., 0.);
        l.isClosed() = false;

        cavc::Polyline<double> circle;
        circle.addVertex(1., 0., 1.);
        circle.addVertex(-1., 0., 1.);
        circle.isClosed() = true;

        print_intersects("cavc: circle-line start-end", curve_intersections(circle, l));
        print_intersects("custom: circle-line start-end", curve_intersections_with_ends(circle, l));
        std::cout << "\n";
    }
    {
        std::cout << "13.\n";

        cavc::Polyline<double> l1;
        l1.addVertex(0., 0., 0.);
        l1.addVertex(1., 0., 0.);
        l1.isClosed() = false;

        cavc::Polyline<double> l2;
        l2.addVertex(0., 0., 0.);
        l2.addVertex(1., 0., 0.);
        l2.isClosed() = false;

        print_intersects("cavc: line-line identical", curve_intersections(l1, l2));
        print_intersects("custom: line-line identical", curve_intersections_with_ends(l1, l2));
        std::cout << "\n";
    }
    {
        std::cout << "10.\n";
        cavc::Polyline<double> circle1;
        circle1.addVertex(1., 0., 1.);
        circle1.addVertex(-1., 0., 1.);
        circle1.isClosed() = true;

        cavc::Polyline<double> circle2;
        circle2.addVertex(1., 0., 1.);
        circle2.addVertex(-1., 0., 1.);
        circle2.isClosed() = true;

        print_intersects("cavc: circle-circle identical", curve_intersections(circle1, circle2));
        print_intersects("custom: circle-circle identical", curve_intersections_with_ends(circle1, circle2));
        std::cout << "\n";
    }

//  input.isClosed() = False;
//
//  // compute the resulting offset polylines, offset = 3
//  std::vector<cavc::Polyline<double>> results = cavc::parallelOffset(input, 3.0);



}