// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#include <iostream>
#include <map>
#include <chrono>
#include <mutex>
#include <thread>
#include <x/vio/vio.h>
#include <ctime>

#include <boost/lockfree/spsc_queue.hpp>


#define MEASURE_FREQ
using namespace x;
using namespace std::chrono_literals;
int seq_frame_counter_ = 0;
int seq_imu_counter_ = 0;

std::deque<Eigen::Vector4d> acc_data_;
std::deque<Eigen::Vector4d> gyro_data_;
std::vector<ImuData> imu_data_;


using DataQueue = boost::lockfree::spsc_queue<rs2::motion_frame, boost::lockfree::capacity<1000>>;
using DataFrameQueue = boost::lockfree::spsc_queue<rs2::frameset, boost::lockfree::capacity<30>>;


// The callback example demonstrates asynchronous usage of the pipeline
int main(int argc, char *argv[]) try {

    const int imu_interpolation_rate = 200;
    double sample_time = 0;
    DataQueue motion_queue;
    DataFrameQueue frame_queue;


    // xVIO instance
    VIO vio;
    std::filesystem::path path("/params.yaml");
    const auto params = vio.loadParamsFromYaml(path);
    vio.setUp(params);
    vio.stopTlio();

//    rs2::log_to_console(RS2_LOG_SEVERITY_INFO);

    std::map<int, int> counters;
    std::map<int, std::string> stream_names;
    std::mutex mutex;

#ifdef MEASURE_FREQ
    clock1 = clock();
    clock3 = clock();
    clock5 = clock();
#endif


    std::atomic_bool imu_sent{false};
    std::atomic_bool start_{true};
    auto thr_motion = std::thread(
            [&start_, &frame_queue, &vio, &imu_sent, &updated_state, &mtx, &sample_time]() {
                while (start_) {
                    if (!vio.isInitialized()) { continue; }
                    if (!imu_sent) { continue; }
                    frame_queue.consume_all([&](const rs2::frameset &fs) {
                        auto c1 = clock();
                        double timestamp = fs.get_timestamp() / 1000;
                        cv::Mat frame = frame_to_mat(fs.get_fisheye_frame(0));
                        TiledImage match_img = TiledImage(frame);

                        TiledImage feature_img = TiledImage(match_img);

                        if ((timestamp -
                             sample_time) < 0) {
                            throw std::runtime_error("PROBLEMS:::::: with time");
                        }
                        if (imu_data_.empty() || imu_data_.back().timestamp < timestamp) { return; }
                        std::cout << "IMU data passed: " << int((timestamp -
                                                                 sample_time) *
                                                                imu_interpolation_rate) << std::endl;
                        std::vector<ImuData> interp_imu = LinearInterpolation::interpolateIMU(imu_data_,
                                                                                              sample_time,
                                                                                              static_cast<double>(1.0 /
                                                                                                                  imu_interpolation_rate),
                                                                                              int((timestamp -
                                                                                                   sample_time) *
                                                                                                  imu_interpolation_rate));

                        for (const auto &imu: interp_imu) {
                            vio.processImu(imu.timestamp, seq_imu_counter_, imu.w, imu.a);
                            seq_imu_counter_++;
                        }
                        imu_data_.erase(std::remove_if(imu_data_.begin(), imu_data_.end(),
                                                       [&](auto &item) {
                                                           if (item.timestamp < timestamp) {
                                                               return true;
                                                           }
                                                       }), imu_data_.end());

                        {
                            std::lock_guard<std::mutex> lck(mtx);
                            updated_state = vio.processImageMeasurement(timestamp, seq_frame_counter_, match_img,
                                                                        feature_img);
                        }
                        seq_frame_counter_++;
                        sample_time = timestamp;
                        cv::imshow("Frame", frame);
                        if (updated_state.has_value()) {
//                    cv::imshow("Matches", match_img);
                            cv::imshow("Feature tracks", feature_img);
                        }
                        cv::waitKey(1);
                        imu_sent = false;

                        auto c2 = clock();
                        auto delta_t = ((double) (c2 - c1) / CLOCKS_PER_SEC);
#ifdef MEASURE_FREQ
                        clock5 = clock();
                        std::cout << "Time taken by FRAME is : "
                                  << 1.0 / ((double) (clock5 - clock6) / CLOCKS_PER_SEC) << std::setprecision(15);
                        std::cout << " Hz " << std::endl;
                        clock6 = clock();
#endif
                        std::cout << std::setprecision(17) << "Frame Process_time: " << (delta_t) << " s" << std::endl;
                        std::this_thread::sleep_for(100us);
//                        std::this_thread::sleep_for(
//                                std::chrono::duration<double, std::milli>(2 - delta_t));
                    });
                    std::this_thread::sleep_for(100us);
                }
            });
    thr_motion.detach();

    auto thr_imu = std::thread([&start_, &motion_queue, &vio, &imu_sent]() {
        while (start_) {
            if (!vio.isInitialized()) { continue; }
            motion_queue.consume_all([&](const rs2::motion_frame &frame) {
                auto c1 = clock();

                if (frame.get_profile().stream_type() == rs2_stream::RS2_STREAM_GYRO) {
                    const rs2_vector v = frame.get_motion_data();
                    gyro_data_.emplace_back(v.x, v.y, v.z, frame.get_timestamp() / 1000);
                } else if (frame.get_profile().stream_type() == rs2_stream::RS2_STREAM_ACCEL) {
                    const rs2_vector v = frame.get_motion_data();
                    acc_data_.emplace_back(v.x, v.y, v.z, frame.get_timestamp() / 1000);

                    interpolateIMU(gyro_data_, acc_data_, vio);
//                      std::cout << "process imu" << std::endl;
//                      auto s = vio.processImu(frame.get_timestamp() / 1000, seq_imu_counter_, {v.x, v.y, v.z},
//                                            {v.x, v.y, v.z});
                    imu_sent = true;
                } else { return; }

                auto c2 = clock();
                auto delta_t = ((double) (c2 - c1) / CLOCKS_PER_SEC);
                std::cout << std::setprecision(17) << "IMU Process_time: " << (delta_t) << " s" << std::endl;
                std::this_thread::sleep_for(100us);
            });
            std::this_thread::sleep_for(100us);
        }
    });
    thr_imu.detach();

    // Create a configuration for configuring the pipeline with a non default profile
    rs2::config cfg;
    // Add streams of gyro and accelerometer to configuration
    cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
    cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F);
    cfg.enable_stream(RS2_STREAM_FISHEYE, 1);
    cfg.enable_stream(RS2_STREAM_FISHEYE, 2);

    rs2::pipeline pipe;
    rs2::pipeline_profile profiles = pipe.start(cfg, [&](const rs2::frame &frame) {
        std::lock_guard<std::mutex> lock(mutex);

        switch (frame.get_profile().stream_type()) {
            case rs2_stream::RS2_STREAM_ACCEL: {
                auto fs = frame.as<rs2::motion_frame>();
                motion_queue.push(fs);
//#ifdef MEASURE_FREQ
//                clock2 = clock();
//                std::cout << "Time taken by ACC is : "
//                          << 1.0 / ((double) (clock2 - clock1) / CLOCKS_PER_SEC) << std::setprecision(15);
//                std::cout << " Hz " << std::endl;
//                clock1 = clock();
//#endif

            }
                break;
            case rs2_stream::RS2_STREAM_GYRO: {
                auto fs = frame.as<rs2::motion_frame>();
                if (!vio.isInitialized()) {
                    sample_time = frame.get_timestamp() / 1000;
                    vio.initAtTime(sample_time);

                    imu_data_.erase(std::remove_if(imu_data_.begin(), imu_data_.end(),
                                                   [&](auto &item) {
                                                       if (item.timestamp < sample_time) {
                                                           return true;
                                                       }
                                                   }), imu_data_.end());
                }
                motion_queue.push(fs);
//#ifdef MEASURE_FREQ
//                clock3 = clock();
//                std::cout << "Time taken by GYRO is : "
//                          << 1.0 / ((double) (clock3 - clock4) / CLOCKS_PER_SEC) << std::setprecision(15);
//                std::cout << " Hz " << std::endl;
//                clock4 = clock();
//#endif

            }
                break;
            case rs2_stream::RS2_STREAM_FISHEYE: {
                auto fs = frame.as<rs2::frameset>();
                frame_queue.push(fs);
//#ifdef MEASURE_FREQ
//                clock5 = clock();
//                std::cout << "Time taken by FRAME is : "
//                          << 1.0 / ((double) (clock5 - clock6) / CLOCKS_PER_SEC) << std::setprecision(15);
//                std::cout << " Hz " << std::endl;
//                clock6 = clock();
//#endif
            }
                break;
            default:
                break;
        }
        std::this_thread::sleep_for(100us);

    });


    while (!pangolin::ShouldQuit()) {
        if (updated_state.has_value()) {
            const Vector3 pos = updated_state->computeCameraPosition();
            const Matrix3 rot = updated_state->computeCameraAttitude().toQuaternion().toRotationMatrix();

            Eigen::Matrix4d m = Eigen::Matrix4d::Identity();
            m.block<3, 3>(0, 0) = rot;
            m(0, 2) = pos.x();
            m(1, 2) = pos.y();
            m(2, 2) = pos.z();

//            std::cout << m << std::endl;
            if (m.hasNaN()) { continue; }
            pangolin::OpenGlMatrix T_wc(m);

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            d_cam.Activate(s_cam);
            // Render OpenGL Frame
            std::lock_guard<std::mutex> lck(mtx);
            pangolin::glDrawAxis(T_wc, 1.0);

            // Swap frames and Process Events
            pangolin::FinishFrame();
        }
    }


    std::cout << "Press a key to stop...";
    int key;
    std::cin >> key;

    start_ = false;
    if (thr_motion.joinable()) {
        thr_motion.join();
    }
    if (thr_imu.joinable()) {
        thr_imu.join();
    }

    return EXIT_SUCCESS;
}
catch (const rs2::error &e) {
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    "
              << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
