// cpp/bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "forman.h"

namespace py = pybind11;

PYBIND11_MODULE(_forman, m) {
    m.doc() = "Fast Forman gradient computation for tree crown segmentation";
    
    // Bind Point struct
    py::class_<forman::Point>(m, "Point")
        .def(py::init<int, int>())
        .def_readwrite("x", &forman::Point::x)
        .def_readwrite("y", &forman::Point::y);
    
    // Bind CellComplex class
    py::class_<forman::CellComplex>(m, "CellComplex")
        // Constructor that takes width, height, and numpy array
        .def(py::init([](int width, int height, py::array_t<double> array) {
            py::buffer_info buf = array.request();
            double* ptr = static_cast<double*>(buf.ptr);
            return std::make_unique<forman::CellComplex>(width, height, ptr);
        }), py::arg("width"), py::arg("height"), py::arg("data"))
        
        // Or simpler: constructor that takes just a numpy array (auto-detect dimensions)
        .def(py::init([](py::array_t<double> array) {
            py::buffer_info buf = array.request();
            if (buf.ndim != 2) {
                throw std::runtime_error("Input must be a 2D array");
            }
            int height = buf.shape[0];
            int width = buf.shape[1];
            double* ptr = static_cast<double*>(buf.ptr);
            return std::make_unique<forman::CellComplex>(width, height, ptr);
        }))
        .def("build", &forman::CellComplex::build)
        .def("get_critical_points", &forman::CellComplex::get_critical_points)
        .def("get_basin", &forman::CellComplex::get_basin)
        .def("filter_by_persistence", &forman::CellComplex::filter_by_persistence)
        .def("get_all_basins", &forman::CellComplex::get_all_basins)
        .def("get_gradient_pairs", &forman::CellComplex::get_gradient_pairs)
        .def("print_statistics", &forman::CellComplex::print_statistics)
        .def_property_readonly("width", &forman::CellComplex::width)
        .def_property_readonly("height", &forman::CellComplex::height)
        .def("num_vertices", &forman::CellComplex::num_vertices)
        .def("get_value", py::overload_cast<int, int>(&forman::CellComplex::get_value, py::const_))
        .def("get_value", py::overload_cast<size_t>(&forman::CellComplex::get_value, py::const_));
}