//
// Created by viciopoli on 02.06.23.
//
#include <x/vio/vio.h>
#include <x/ekf/updater.h>
#include <pybind11/pybind11.h>
#include "../external/pybind11cv/ndarray_converter.h"


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

//namespace x {
//    class PyCameraModel : public x::CameraModel {
//    public:
//        /* Inherit the constructors */
//        using CameraModel::CameraModel;
//
//        /* Trampoline (need one for each virtual function) */
//        void undistort(Feature &feature) override {
//            PYBIND11_OVERRIDE_PURE(
//                    void, /* Return type */
//                    x::CameraModel,      /* Parent class */
//                    undistort,          /* Name of function in C++ (must match Python name) */
//                    feature      /* Argument(s) */
//            );
//        }
//    };
//}

// namespace pybind11::detail
PYBIND11_MODULE(x_bind, m) {
    m.attr("__version__") = "1.3.3";

    NDArrayConverter::init_numpy();

    py::class_<std::filesystem::path>(m, "Path").def(py::init<std::string>(), py::arg("path"));
    py::implicitly_convertible<std::string, std::filesystem::path>();

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
            .def("timestamp", &Feature::getTimestamp)
            .def(py::pickle(
                    [](const Feature &p) { // __getstate__
                        /* Return a tuple that fully encodes the state of the object */
                        return py::make_tuple(p.getTimestamp(), p.getX(), p.getY(), p.getXDist(), p.getYDist());
                    },
                    [](py::tuple t) { // __setstate__
                        if (t.size() != 5)
                            throw std::runtime_error("Invalid state!");

                        /* Create a new C++ instance */
                        Feature f(t[0].cast<double>(),
                                  0,
                                  t[1].cast<double>(),
                                  t[2].cast<double>(),
                                  t[3].cast<double>(),
                                  t[4].cast<double>(),
                                  0);
                        return f;
                    }
            ));

    py::class_<Match>(m, "Match")
            .def(py::init<Feature, Feature>(), "previous"_a, "current"_a)
            .def_readwrite("previous", &Match::previous)
            .def_readwrite("current", &Match::current)
            .def(py::pickle(
                    [](const Match &p) { // __getstate__
                        /* Return a tuple that fully encodes the state of the object */
                        return py::make_tuple(p.previous, p.current);
                    },
                    [](py::tuple t) { // __setstate__
                        if (t.size() != 2)
                            throw std::runtime_error("Invalid state!");

                        /* Create a new C++ instance */
                        Match m;
                        m.previous = t[0].cast<Feature>();
                        m.current = t[1].cast<Feature>();
                        return m;
                    }));

    py::bind_vector<x::MatchList>(m, "MatchList");

//    py::class_<MatchList>(m, "MatchList")
//            .def(py::init<>())
//            .def("clear", &MatchList::clear)
//            .def("pop_back", &MatchList::pop_back)
//            .def("append",&MatchList::push_back)
//            .def("__len__", [](const MatchList &v) { return v.size(); })
//            .def("__iter__", [](MatchList &v) {
//                return py::make_iterator(v.begin(), v.end());
//            }, py::keep_alive<0, 1>());

    py::class_<State>(m, "State")
            .def(py::init<>())
            .def("getPosition", &State::getPosition)
            .def("getOrientation", [](State &self) -> std::vector<double> {
                Eigen::Quaterniond q = self.getOrientation();
                return {q.w(), q.x(), q.y(), q.z()};
            });

    py::class_<Updater, VioUpdater>(m, "VioUpdater")
            .def(py::init<>());

    py::class_<VIO>(m, "VIO")
            .def(py::init<>())
            .def("initAtTime", &VIO::initAtTime, "time"_a)
            .def("isInitialized", &VIO::isInitialized, "True if initialized")
            .def("isInitialized", &VIO::isInitialized, "Return true if initialized")
            .def("processImu", &VIO::processImu, "timestamp"_a, "seq"_a, "w_m"_a, "a_m"_a)
            .def("processTracksNoFrame", &VIO::processTracksNoFrame, "timestamp"_a, "seq"_a,
                 "matches"_a, "h"_a, "w"_a, py::return_value_policy::copy)
            .def("processTracks", [](VIO &self, const double &timestamp,
                                     const unsigned int seq, const MatchList &matches, const cv::Mat &image) {
                     auto p = self.getParams();
                     TiledImage match_img = TiledImage(image, timestamp,
                                                       seq, p.n_tiles_h, p.n_tiles_w,
                                                       p.max_feat_per_tile);
                     TiledImage feature_img = TiledImage(match_img.clone());
                     self.processTracks(timestamp, seq, matches, match_img, feature_img);
                     cv::Mat return_img(feature_img);
                     return return_img;
                 }, "timestamp"_a, "seq"_a,
                 "matches"_a, "image"_a)
            .def("setUp", &VIO::setUp, "params"_a)
            .def("loadParamsFromYaml",
                 [](VIO &self, std::string &path) {
                     auto a = fsm::path(path);
                     return self.loadParamsFromYaml(a);
                 }, "path"_a);


    py::class_<Camera::Params>(m, "CamParams")
            .def(py::init<>())
            .def("inv_fx", [](Camera::Params &self) {
                return self.inv_fx_;
            })
            .def("inv_fy", [](Camera::Params &self) {
                return self.inv_fy_;
            })
            .def("fx", [](Camera::Params &self) {
                return self.fx_;
            })
            .def("fy", [](Camera::Params &self) {
                return self.fy_;
            })
            .def("cx_n", [](Camera::Params &self) {
                return self.cx_n_;
            })
            .def("cy_n", [](Camera::Params &self) {
                return self.cy_n_;
            })
            .def("cx", [](Camera::Params &self) {
                return self.cx_;
            })
            .def("cy", [](Camera::Params &self) {
                return self.cy_;
            })
            .def("dist_coeff", [](Camera::Params &self) {
                return self.dist_coeff;
            });

//    py::class_<CameraModel, PyCameraModel>(m, "CameraModel")
//            .def(py::init<Camera::Params &>(), "params"_a)
//            .def("undistort", [](CameraModel &self, Feature &f) {
//                self.undistort(f);
//                return f;
//            });


//    py::class_<CameraFov>(m, "CameraFov")
//            .def(py::init<Camera::Params &>(), "params"_a)
//            .def("undistort", &CameraFov::undistort);
//
    py::class_<Params>(m, "Params")
            .def(py::init<>())
            .def("camera_params", [](Params &self, Camera::Params &p) {
                p = self.camera->getCameraParams();
            });
}
