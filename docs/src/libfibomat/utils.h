#ifndef LIBFIBOMAT_UTILS_H
#define LIBFIBOMAT_UTILS_H

// #include <range/v3/all.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <cavc/vector2.hpp>

namespace fibomat
{
    struct point_for_conversion
    {
        double point[3];

        double operator[](std::size_t i) { return point[i]; }
    };
    auto py_array_to_points(py::array_t<double> &&curve_points)
    {
        py::buffer_info info = curve_points.request();
        if (info.ndim != 2)
        {
            throw std::runtime_error("curve_points must be a 2d array.");
        }

        if (info.shape[1] != 3)
        {
            throw std::runtime_error("curve_points must have 3 elements second axis.");
        }

        auto raw_points = curve_points.mutable_unchecked<2>();

        std::vector<std::tuple<double, double, double>> points;
        // std::vector<point_for_conversion> points;

        for (int i = 0; i < info.shape[0]; ++i)
        {
            points.emplace_back(std::make_tuple(raw_points(i, 0), raw_points(i, 1), raw_points(i, 2)));
            // points.emplace_back({&raw_points(3*i)});
        }

        //        auto points =
        //                ranges::span<double>{raw_points.mutable_data(0, 0), raw_points.size()} | ranges::views::chunk(3);

        return points;
    }

    // template<typename T> class iterator_t : public py::iterator {
    // public:
    //     using iterator::iterator;

    //     // cheap/lazy way of doing this as it could be done in advance(),
    //     // but then py::iterator needs to be changed
    //     T operator*() const {
    //         return this->iterator::operator*().template cast<T>();
    //     }
    // };

    // template<typename T> class iterable_t : public py::iterable {
    // public:
    //     using iterable::iterable;

    //     iterator_t<T> begin() { return py::reinterpret_borrow<py::iterator>(PyObject_GetIter(ptr())); } // { return {PyObject_GetIter(ptr()), false}; }
    //     iterator_t<T> end() { return py::reinterpret_borrow<py::iterator>(nullptr);}// { return {nullptr, false}; }
    // };

    template <typename T>
    cavc::Vector2<T> iterable_to_vector(py::iterable iterable)
    {
        // https://github.com/pybind/pybind11/issues/1773
        auto vec = iterable.cast<std::vector<T>>();

        if (vec.size() != 2)
        {
            throw std::runtime_error("Cannot construct Vector2 from iterable with size != 2");
        }
        return cavc::Vector2<T>(vec[0], vec[1]);
    }

}

#endif // LIBFIBOMAT_UTILS_H
