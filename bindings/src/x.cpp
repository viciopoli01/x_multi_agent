//
// Created by viciopoli on 02.06.23.
//
#include <x/vio/vio.h>
#include <x/ekf/updater.h>
#include <pybind11/pybind11.h>

#include <pybind11/chrono.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <filesystem>

using namespace pybind11::literals;
namespace py = pybind11;
using namespace x;

PYBIND11_MAKE_OPAQUE(x::MatchList)

// namespace pybind11::detail
PYBIND11_MODULE(x_bind, m) {

    // Feature class
    py::class_<Feature>(m, "Feature")
            .def(py::init<const double &, double, double, double>(), "timestamp"_a, "x"_a, "y"_a, "intensity"_a)
            .def(py::init<const double &, unsigned int, double, double, double, double, double>(), "timestamp"_a,
                 "frame_number"_a, "x"_a,
                 "y"_a, "x_dist"_a, "y_dist"_a, "intensity"_a)
            .def_property("x", &Feature::getX, &Feature::setX)
            .def_property("y", &Feature::getY, &Feature::setY)
            .def_property("x_dist", &Feature::getXDist, &Feature::setXDist)
            .def_property("y_dist", &Feature::getYDist, &Feature::setYDist)
            .def("timestamp", &Feature::getTimestamp);

    py::class_<Match>(m, "Match")
            .def(py::init<Feature, Feature>(), "previous"_a, "current"_a)
            .def_readwrite("previous", &Match::previous)
            .def_readwrite("current", &Match::current);

    py::bind_vector<x::MatchList>(m, "MatchList");


    py::class_<State>(m, "State")
            .def(py::init<>())
            .def("getPosition", &State::getPosition)
            .def("getOrientation", &State::getOrientation);

    py::class_<Updater, VioUpdater>(m, "VioUpdater")
            .def(py::init<>());

    py::class_<VIO>(m, "VIO")
            .def(py::init<>())
            .def("initAtTime", &VIO::initAtTime, "time"_a)
            .def("isInitialized", &VIO::isInitialized, "True if initialized")
            .def("isInitialized", &VIO::isInitialized, "Return true if initialized")
            .def("processImu", &VIO::processImu, "timestamp"_a, "seq"_a, "w_m"_a, "a_m"_a)
            .def("processTracksNoFrame", &VIO::processTracksNoFrame, "timestamp"_a, "seq"_a,
                 "matches"_a, "h"_a, "w"_a, py::return_value_policy::reference_internal)
            .def("setUp", &VIO::setUp, "params"_a)
            .def("loadParamsFromYaml",
                 [](VIO &self, std::string &path) {
                     auto a = fsm::path(path);
                     return self.loadParamsFromYaml(a);
                 }, "path"_a);

    py::class_<Params>(m, "Params")
            .def(py::init<>());
}
