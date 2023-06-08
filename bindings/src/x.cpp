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
PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector3d>)

namespace x_py {
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
    template<class CameraModelBase = x::CameraModel>
    class PyCameraModel : public CameraModelBase {
    public:
        using CameraModelBase::CameraModelBase; // Inherit constructors
        void undistort(x::Feature &feature) override {
            PYBIND11_OVERRIDE_PURE(void, CameraModelBase, undistort, feature);
        }
    };


//    class PyCameraFov : public x::CameraFov {
//    public:
//        /* Inherit the constructors */
//        using CameraFov::CameraFov;
//
//        /* Trampoline (need one for each virtual function) */
//        void undistort(Feature &feature) override {
//            PYBIND11_OVERRIDE_PURE(
//                    void, /* Return type */
//                    x::CameraFov,      /* Parent class */
//                    undistort,          /* Name of function in C++ (must match Python name) */
//                    feature      /* Argument(s) */
//            );
//        }
//    };
//
//    class PyCameraRadTan : public x::CameraRadTan {
//    public:
//        /* Inherit the constructors */
//        using CameraRadTan::CameraRadTan;
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
//
//    class PyCameraEquidistant : public x::CameraEquidistant {
//    public:
//        /* Inherit the constructors */
//        using CameraEquidistant::CameraEquidistant;
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
}

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

    py::class_<State>(m, "State")
            .def(py::init<>())
            .def("getPosition", &State::getPosition)
            .def("getOrientation", [](State &self) -> std::vector<double> {
                Eigen::Quaterniond q = self.getOrientation();
                return {q.w(), q.x(), q.y(), q.z()};
            })
            .def("computeCameraPosition", &State::computeCameraPosition, "Return camera Position")
            .def("computeCameraOrientation", [](State &self) -> std::vector<double> {
                Eigen::Quaterniond q = self.computeCameraOrientation();
                return {q.w(), q.x(), q.y(), q.z()};
            }, "Return camera Orientation");

    py::class_<Updater, VioUpdater>(m, "VioUpdater")
            .def(py::init<>());

    py::bind_vector<std::vector<Eigen::Vector3d>>(m, "Vector3d");

    py::class_<VIO>(m, "VIO")
            .def(py::init<>())
            .def("initAtTime", &VIO::initAtTime, "time"_a)
            .def("isInitialized", &VIO::isInitialized, "True if initialized")
            .def("isInitialized", &VIO::isInitialized, "Return true if initialized")
            .def("processImu", &VIO::processImu, "timestamp"_a, "seq"_a, "w_m"_a, "a_m"_a)
            .def("computeSLAMCartesianFeaturesForState", &VIO::computeSLAMCartesianFeaturesForState, "state"_a)
            .def("processTracks", [](VIO &self, const double &timestamp,
                                     const unsigned int seq, const MatchList &matches, const cv::Mat &image) {
                     auto p = self.getParams();
                     TiledImage match_img = TiledImage(image, timestamp,
                                                       seq, p.n_tiles_h, p.n_tiles_w,
                                                       p.max_feat_per_tile);
                     TiledImage feature_img = TiledImage(match_img.clone());
                     auto s = self.processTracks(timestamp, seq, matches, match_img, feature_img);
                     cv::Mat return_img(feature_img);
                     return py::make_tuple(s, return_img);
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

    py::class_<CameraModel, std::shared_ptr<CameraModel>, x_py::PyCameraModel<>>(m, "CameraModel")
            .def(py::init<const Camera::Params &>(), "params"_a);

    py::class_<CameraFov, std::shared_ptr<CameraFov>, CameraModel, x_py::PyCameraModel<CameraFov>>(m, "CameraFov")
            .def(py::init<const Camera::Params &>(), "params"_a)
            .def("undistort", &CameraFov::undistort, "feature"_a)
            .def("normalize", [](CameraFov &self, const Feature &f) { return self.normalize(f); }, "feature"_a)
            .def("normalize",
                 [](CameraFov &self, const Track &t, size_t max_size = 0) { return self.normalize(t, max_size); },
                 "track"_a, "max_size"_a)
            .def("normalize",
                 [](CameraFov &self, const TrackList &t, size_t max_size = 0) { return self.normalize(t, max_size); },
                 "track_list"_a, "max_size"_a);

    py::class_<CameraRadTan, std::shared_ptr<CameraRadTan>, CameraModel, x_py::PyCameraModel<CameraRadTan>>(m,
                                                                                                            "CameraRadTan")
            .def(py::init<const Camera::Params &>(), "params"_a)
            .def("undistort", &CameraRadTan::undistort, "feature"_a)
            .def("normalize", [](CameraRadTan &self, const Feature &f) { return self.normalize(f); }, "feature"_a)
            .def("normalize",
                 [](CameraRadTan &self, const Track &t, size_t max_size = 0) { return self.normalize(t, max_size); },
                 "track"_a, "max_size"_a)
            .def("normalize",
                 [](CameraRadTan &self, const TrackList &t, size_t max_size = 0) {
                     return self.normalize(t, max_size);
                 },
                 "track_list"_a, "max_size"_a);

    py::class_<CameraEquidistant, std::shared_ptr<CameraEquidistant>, CameraModel, x_py::PyCameraModel<CameraEquidistant>>(
            m, "CameraEquidistant")
            .def(py::init<const Camera::Params &>(), "params"_a)
            .def("undistort", &CameraEquidistant::undistort, "feature"_a).def("normalize", [](CameraFov &self,
                                                                                              const Feature &f) {
                return self.normalize(f);
            }, "feature"_a)
            .def("normalize", [](CameraEquidistant &self, const Feature &f) { return self.normalize(f); }, "feature"_a)
            .def("normalize",
                 [](CameraEquidistant &self, const Track &t, size_t max_size = 0) {
                     return self.normalize(t, max_size);
                 },
                 "track"_a, "max_size"_a)
            .def("normalize",
                 [](CameraEquidistant &self, const TrackList &t, size_t max_size = 0) {
                     return self.normalize(t, max_size);
                 },
                 "track_list"_a, "max_size"_a);


    py::class_<Params>(m, "Params")
            .def(py::init<>())
            .def("camera_params", [](Params &self, Camera::Params &p) {
                p = self.camera->getCameraParams();
            })
            .def("non_max_supp", [](const Params &self) { return self.non_max_supp; })
            .def("block_half_length", [](const Params &self) { return self.block_half_length; })
            .def("margin", [](const Params &self) { return self.margin; })
            .def("n_feat_min", [](const Params &self) { return self.n_feat_min; })
            .def("outlier_method", [](const Params &self) { return self.outlier_method; })
            .def("outlier_param1", [](const Params &self) { return self.outlier_param1; })
            .def("outlier_param2", [](const Params &self) { return self.outlier_param2; })
            .def("win_size_w", [](const Params &self) { return self.win_size_w; })
            .def("win_size_h", [](const Params &self) { return self.win_size_h; })
            .def("max_level", [](const Params &self) { return self.max_level; })
            .def("min_eig_thr", [](const Params &self) { return self.min_eig_thr; })
            .def("n_tiles_h", [](const Params &self) { return self.min_eig_thr; })
            .def("n_tiles_h", [](const Params &self) { return self.min_eig_thr; })
            .def("fast_detection_delta", [](const Params &self) { return self.fast_detection_delta; })
            .def("camera", [](const Params &self) {
                return self.camera;
            });

    py::class_<Tracker>(m, "Tracker")
            .def(py::init<>())
            .def("set_params",
                 [](Tracker &self, std::shared_ptr<CameraModel> &cam, int fast_detection_delta, bool non_max_supp,
                    unsigned int block_half_length, unsigned int margin,
                    unsigned int n_feat_min, int outlier_method,
                    double outlier_param1, double outlier_param2, int win_size_w,
                    int win_size_h, int max_level, double min_eig_thr) {
                     self.setParams(cam, fast_detection_delta, non_max_supp, block_half_length, margin, n_feat_min,
                                    outlier_method,
                                    outlier_param1, outlier_param2, win_size_w,
                                    win_size_h, max_level, min_eig_thr);
                 }, "cam"_a, "fast_detection_delta"_a, "non_max_supp"_a,
                 "block_half_length"_a, "margin"_a,
                 "n_feat_min"_a, "outlier_method"_a,
                 "outlier_param1"_a, "outlier_param2"_a, "win_size_w"_a,
                 "win_size_h"_a, "max_level"_a, "min_eig_thr"_a)
            .def("get_matches", &Tracker::getMatches, "Retrieve matches")
            .def("track", [](Tracker &self, cv::Mat &current_img, const int seq, const double &timestamp,
                             unsigned int frame_number, const int n_tiles_h, const int n_tiles_w,
                             const int max_feat_per_tile) {
                     TiledImage current_img_tiled = TiledImage(current_img, timestamp,
                                                               seq, n_tiles_h, n_tiles_w,
                                                               max_feat_per_tile);
                     self.track(current_img_tiled, timestamp, frame_number);
                 }, "current_img"_a, "seq"_a, "timestamp"_a, "frame_number"_a, "n_tiles_h"_a, "n_tiles_w"_a,
                 "max_feat_per_tile"_a);
}
