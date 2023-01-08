#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Image.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <x/vio/vio.h>
#include <ctime>

#include <cv_bridge/cv_bridge.h>
#include <boost/lockfree/spsc_queue.hpp>


#define IMURATE 200
#define FRAMERATE 30

using namespace x;
using namespace std::chrono_literals;
namespace enc = sensor_msgs::image_encodings;

using ImageDataQueue = boost::lockfree::spsc_queue<sensor_msgs::Image::ConstPtr, boost::lockfree::capacity<10000>>;
using ImuDataQueue = boost::lockfree::spsc_queue<sensor_msgs::Imu::ConstPtr, boost::lockfree::capacity<10000>>;


// The callback example demonstrates asynchronous usage of the pipeline
int main(int argc, char *argv[]) {
    // xVIO instance
    VIO vio;
    std::filesystem::path path("UAV0_params_thermal.yaml");
    const auto params = vio.loadParamsFromYaml(path);
    vio.setUp(params);

    // Store data
    std::ofstream data_results;
    data_results.open("example.txt");

    // Start threads
    ImageDataQueue imageDataQueue;
    ImuDataQueue imuDataQueue;
    std::atomic_bool start = true;
    auto imu_thr = std::thread([&]() {
        while (start) {
            imuDataQueue.consume_all([&](const sensor_msgs::Imu::ConstPtr &ptr) {
                if (!vio.isInitialized()) {
                    vio.initAtTime(ptr->header.stamp.toSec());
                }
                // Read accels
                Vector3 a_m(ptr->linear_acceleration.x,
                            ptr->linear_acceleration.y,
                            ptr->linear_acceleration.z);

                // Read gyros
                Vector3 w_m(ptr->angular_velocity.x,
                            ptr->angular_velocity.y,
                            ptr->angular_velocity.z);

                // Call xVIO IMU propagation
                const auto propagated_state = vio.processImu(ptr->header.stamp.toSec(),
                                                             ptr->header.seq, w_m, a_m);
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            });
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });
    imu_thr.detach();


    // Set up tracker module
    Tracker tracker(params.camera, params.fast_detection_delta, params.non_max_supp,
                    params.block_half_length, params.margin, params.n_feat_min,
                    params.outlier_method, params.outlier_param1,
                    params.outlier_param2, params.win_size_w, params.win_size_h,
                    params.max_level, params.min_eig_thr
#ifdef PHOTOMETRIC_CALI
            ,
                    params.temporal_params_div, params.spatial_params,
                    params.spatial_params_thr, params.epsilon_gap,
                    params.epsilon_base, params.max_level_photo,
                    params.min_eig_thr_photo, params.win_size_w_photo,
                    params.win_size_h_photo, params.fast_detection_delta_photo
#endif
#ifdef MULTI_UAV
            , params.descriptor_scale_factor, params.descriptor_patch_size,
                    params.descriptor_pyramid
#endif
    );


    auto image_thr = std::thread([&]() {
        while (start) {
            imageDataQueue.consume_all([&](const sensor_msgs::Image::ConstPtr &ptr) {
                const unsigned int frame_number = ptr->header.seq;
                cv_bridge::CvImageConstPtr cv_ptr;
                try {
                    cv_ptr = cv_bridge::toCvShare(ptr, enc::MONO8);
                } catch (cv_bridge::Exception &e) {
                    std::cerr << "cv_bridge exception: " << e.what() << std::endl;
                    start = false;
                    return;
                }
                cv::Mat img = cv_ptr->image.clone();

                // Shallow copies
                TiledImage match_img = TiledImage(img, ptr->header.stamp.toSec(),
                                                  frame_number, params.n_tiles_h, params.n_tiles_w,
                                                  params.max_feat_per_tile);

                TiledImage feature_img = TiledImage(match_img.clone());

                tracker.track(match_img, ptr->header.stamp.toSec(), frame_number);
                auto matches = tracker.getMatches();

                // Pass matches to VIO
                const auto updated_state = vio.processTracks(ptr->header.stamp.toSec(), frame_number, matches,
                                                             match_img,
                                                             feature_img);

                if (updated_state.has_value()) {
                    cv::imshow("Tracks", feature_img);
                    cv::waitKey(1);
                    auto pos = updated_state->getPosition();
                    data_results << "" << pos.x() << ", " << pos.y() << ", " << pos.z() << std::endl;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            });
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });
    image_thr.detach();

    // Read bag file

    rosbag::Bag bag;
    bag.open("/home/viciopoli/JPL/bags/originalMarsYard/mars_yard_duo_tf.bag", rosbag::bagmode::Read);

    std::vector<std::string> topics;
    topics.emplace_back("/UAV0/image_raw");
    topics.emplace_back("/UAV0/imu");

    for (rosbag::MessageInstance const m: rosbag::View(bag)) {
        if (m.getTopic() == "/UAV0/image_raw") {
            sensor_msgs::Image::ConstPtr s = m.instantiate<sensor_msgs::Image>();
            if (s != nullptr) {
                imageDataQueue.push(s);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        } else if (m.getTopic() == "/UAV0/imu") {
            sensor_msgs::Imu::ConstPtr s = m.instantiate<sensor_msgs::Imu>();
            if (s != nullptr) {
                imuDataQueue.push(s);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    bag.close();
    data_results.close();

    std::cout << "Press Enter to stop..." << std::endl;
    std::cin.ignore();

    start = false;
    if (imu_thr.joinable()) {
        imu_thr.join();
    }
    if (image_thr.joinable()) {
        image_thr.join();
    }

    return 0;
}