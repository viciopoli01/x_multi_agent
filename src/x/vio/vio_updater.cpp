/*
 * Copyright 2020 California  Institute  of Technology (“Caltech”)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "x/vio/vio_updater.h"

#include <iostream>

#include "x/vio/msckf_slam_update.h"
#include "x/vio/msckf_update.h"
#include "x/vio/range_update.h"
#include "x/vio/solar_update.h"
#include "x/vision/types.h"

using namespace x;

VioUpdater::VioUpdater(StateManager &state_manager,
                       TrackManager &track_manager, const double sigma_img,
                       const double sigma_range, const double rho_0,
                       const double sigma_rho_0, const int min_track_length,
#ifdef MULTI_UAV
        std::shared_ptr<PlaceRecognition> place_recognition,
#endif
                       const double sigma_landmark, const double ci_msckf_w,
                       const double ci_slam_w, const int iekf_iter
)
        : state_manager_(state_manager),
          track_manager_(track_manager),
          sigma_img_{sigma_img},
          sigma_landmark_{sigma_landmark},
          sigma_range_{sigma_range},
          rho_0_{rho_0},
          sigma_rho_0_{sigma_rho_0},
          min_track_length_{min_track_length},
          ci_msckf_w_{ci_msckf_w},
          ci_slam_w_(ci_slam_w)
#ifdef MULTI_UAV
,
place_recognition_(place_recognition)
#endif
{
    iekf_iter_ = iekf_iter;
}

double VioUpdater::getTime() const { return measurement_.timestamp; }

TiledImage &VioUpdater::getMatchImage() { return match_img_; }

TiledImage &VioUpdater::getFeatureImage() { return feature_img_; }


#ifdef MULTI_UAV

void VioUpdater::getSlamTracks(TrackList &tracks, std::vector<int> &anchor_idxs,
                               const int n_poses_max) {
  tracks = track_manager_.normalizeSlamTracks(n_poses_max);
  anchor_idxs = state_manager_.getAnchorIdxs();
}

void VioUpdater::getOppTracks(TrackList &tracks) {
  tracks = track_manager_.getOppTracks();
}

void VioUpdater::getMsckfTracks(TrackList &tracks) {
  tracks = track_manager_.getMsckfTracks();
}

void VioUpdater::getSlamTracks(TrackList &tracks) {
  tracks = track_manager_.normalizeSlamTracks(n_poses_max_);
}

bool VioUpdater::preUpdateCI() {
  // PreUpdateCI is used in case of SLAM-SLAM matches
  return !slam_matches_.empty();
}

void VioUpdater::constructSlamCIUpdate(
    const State &state, std::vector<std::shared_ptr<Matrix>> &S_list,
    std::vector<std::shared_ptr<Matrix>> &P_list,
    std::vector<std::shared_ptr<Matrix>> &H_list,
    std::vector<std::shared_ptr<Matrix>> &res_list) {
  // Construct list of pose states (the pose window has been slid
  // and includes current camera pose)
  const TranslationList G_p_C =
      state_manager_.convertCameraPositionsToList(state);
  const AttitudeList C_q_G = state_manager_.convertCameraAttitudesToList(state);

  // Retrieve state covariance prior
  Matrix P = state.getCovariance();
  // Construct update
  const int n_poses_max = state.nPosesMax();

  // Get SLAM feature priors and inverse-depth pose anchors
  const Matrix &feature_states = state.getFeatureArray();
  const std::vector<int> anchor_idxs = state_manager_.getAnchorIdxs();

  // Compute the Jacobians, Covariance, and all the data needed by the KF
  MultiSlamUpdate multi_slam_(slam_trks_, C_q_G, G_p_C, feature_states,
                              anchor_idxs, P, n_poses_max, sigma_landmark_,
                              slam_matches_, ci_slam_w_);

  // Clean the SLAM-SLAM matches from the place recognition
  place_recognition_->cleanSlamMatches();

  // store the computed Jac, Residuals, Innovation, and covariance (Prior cov
  // weighted)
  S_list = multi_slam_.getMultiUAVS();
  P_list = multi_slam_.getMultiUAVCovariance();
  H_list = multi_slam_.getMultiUAVJacobians();
  res_list = multi_slam_.getMultiUAVResiduals();
}
#endif

Vector3dArray VioUpdater::getMsckfInliers() const { return msckf_inliers_; }

Vector3dArray VioUpdater::getMsckfOutliers() const { return msckf_outliers_; }

void VioUpdater::setMeasurement(const VioMeasurement &measurement) {
    measurement_ = measurement;
}

void VioUpdater::preProcess(const State &state) {
    // Construct list of camera orientation states.
    // Note: pose window has not been slid yet and only goes up to
    // the previous frame. We need to crop the first pose out of the
    // list and add the current pose manually.
    // This is used to used check MSCKF track baseline.
    const int n_poses_max = state.nPosesMax();
    const int list_sz = n_poses_max - 1;
    AttitudeList cam_rots =
            state_manager_.convertCameraAttitudesToList(state, list_sz);
    const Attitude last_rot = state.computeCameraAttitude();
    cam_rots.push_back(last_rot);

    // Sort matches into tracks
    feature_img_ = measurement_.image.clone();
    const int n_slam_features_max = state.nFeaturesMax();

#ifdef MULTI_UAV
    // store the Opportunistic tracks that have been matched
    OppIDListPtr opp_ids = place_recognition_->getOppIds();
    track_manager_.setOppUpgradesMSCKF(opp_ids);
#endif

    // update the tracks and promote the tracks that have been matched in the
    // MULTI_UAV case
    track_manager_.manageTracks(measurement_.matches, cam_rots, n_poses_max,
                                n_slam_features_max, min_track_length_,
                                feature_img_);

    // Get image measurements
    slam_trks_ = track_manager_.normalizeSlamTracks(n_poses_max);
    msckf_trks_ = track_manager_.getMsckfTracks();
    msckf_short_trks_ = track_manager_.getShortMsckfTracks();
    new_slam_std_trks_ = track_manager_.getNewSlamStdTracks();
    new_msckf_slam_trks_ = track_manager_.getNewSlamMsckfTracks();

    // Collect indexes of persistent tracks that were lost at this time step
    lost_slam_trk_idxs_ = track_manager_.getLostSlamTrackIndexes();

#ifdef MULTI_UAV
    // update opp matches to MSCKF/SLAM matches
    TrackList opp_tmp = track_manager_.getOppTracks();
    // get the MSCKF matches
    msckf_matches_ = place_recognition_->getMsckfMatches();
    place_recognition_->updateOppMatches(msckf_trks_, slam_trks_, opp_tmp);
#endif

// #ifdef PHOTOMETRIC_CALI
//     // TODO fix data race (?)
//     TrackList all_tracks = TrackList();
//     TrackList opp_tracks = track_manager_.getOppTracks();
//     all_tracks.insert(all_tracks.end(), slam_trks_.begin(), slam_trks_.end());
//     all_tracks.insert(all_tracks.end(), msckf_trks_.begin(), msckf_trks_.end());
//     all_tracks.insert(all_tracks.end(), opp_tracks.begin(), opp_tracks.end());
//     place_recognition_->setIntensistyHistory(all_tracks);
// #endif
}

bool VioUpdater::preUpdate(State &state) {
    // Manage vision states to be added, removed, reparametrized or slid
    state_manager_.manage(state, lost_slam_trk_idxs_);

    // Return true if there are any visual update measurements
    return !(msckf_trks_.empty() && slam_trks_.empty() &&
             new_slam_std_trks_.empty() && new_msckf_slam_trks_.empty());
}

bool VioUpdater::preUpdateShortMsckf() {
#ifdef MULTI_UAV
    // get the SLAM matches
    slam_matches_ = place_recognition_->getSlamMatches();
#endif
    return !msckf_short_trks_.empty();
}

#ifdef MULTI_UAV
void VioUpdater::constructShortMsckfUpdate(
    const State &state, Matrix &h, Matrix &res, Matrix &r,
    std::vector<std::shared_ptr<Matrix>> &S_list,
    std::vector<std::shared_ptr<Matrix>> &P_list,
    std::vector<std::shared_ptr<Matrix>> &H_list,
    std::vector<std::shared_ptr<Matrix>> &res_list) {
#else

void VioUpdater::constructShortMsckfUpdate(const State &state, Matrix &h,
                                           Matrix &res, Matrix &r) {
#endif
    // Construct list of pose states (the pose window has been slid
    // and includes current camera pose)
    const TranslationList G_p_C =
            state_manager_.convertCameraPositionsToList(state);
    const AttitudeList C_q_G = state_manager_.convertCameraAttitudesToList(state);

    // Retrieve state covariance prior
    Matrix P = state.getCovariance();
    // Construct update
    const int n_poses_max = state.nPosesMax();

    /* MSCKF */
#ifdef MULTI_UAV
    MsckfUpdate msckf(msckf_short_trks_, C_q_G, G_p_C, P, n_poses_max, sigma_img_,
                      msckf_matches_, ci_msckf_w_);

    // store the Multi MSCKF CI_EKF information
    S_list = msckf.getMultiUAVS();
    P_list = msckf.getMultiUAVCovariance();
    H_list = msckf.getMultiUAVJacobians();
    res_list = msckf.getMultiUAVResiduals();
#else
    MsckfUpdate msckf(msckf_short_trks_, C_q_G, G_p_C, P, n_poses_max,
                      sigma_img_);
#endif

    // Get matrices
    res = msckf.getResidual();
    h = msckf.getJacobian();
    const Eigen::VectorXd &r_msckf_diag = msckf.getCovDiag();

    /* Combined update */
    r = r_msckf_diag.asDiagonal();

    // QR decomposition of the update Jacobian
    applyQRDecomposition(h, res, r);
}

#ifdef MULTI_UAV
void VioUpdater::constructUpdate(
    const State &state, Matrix &h, Matrix &res, Matrix &r,
    std::vector<std::shared_ptr<Matrix>> &S_list,
    std::vector<std::shared_ptr<Matrix>> &P_list,
    std::vector<std::shared_ptr<Matrix>> &H_list,
    std::vector<std::shared_ptr<Matrix>> &res_list) {
#else

void VioUpdater::constructUpdate(const State &state, Matrix &h, Matrix &res,
                                 Matrix &r) {
#endif
    // Construct list of pose states (the pose window has been slid
    // and includes current camera pose)
    const TranslationList G_p_C =
            state_manager_.convertCameraPositionsToList(state);
    const AttitudeList C_q_G = state_manager_.convertCameraAttitudesToList(state);

    // Retrieve state covariance prior
    Matrix P = state.getCovariance();
    // Construct update
    const int n_poses_max = state.nPosesMax();

    // Set up triangulation for MSCKF // TODO(jeff) set from params
    const unsigned int max_iter = 10;  // Gauss-Newton max number of iterations
    const double term = 0.00001;       // Gauss-Newton termination criterion
    const Triangulation triangulator(C_q_G, G_p_C, max_iter, term);

    /* MSCKF */
#ifdef MULTI_UAV
    MsckfUpdate msckf(msckf_trks_, C_q_G, G_p_C, P, n_poses_max, sigma_img_,
                      msckf_matches_, ci_msckf_w_);

    // store the Multi MSCKF CI_EKF information
    S_list = msckf.getMultiUAVS();
    P_list = msckf.getMultiUAVCovariance();
    H_list = msckf.getMultiUAVJacobians();
    res_list = msckf.getMultiUAVResiduals();
#else
    MsckfUpdate msckf(msckf_trks_, C_q_G, G_p_C, P, n_poses_max, sigma_img_);
#endif

    // Get matrices
    const Matrix &res_msckf = msckf.getResidual();
    const Matrix &h_msckf = msckf.getJacobian();
    const Eigen::VectorXd &r_msckf_diag = msckf.getCovDiag();
    const Vector3dArray &msckf_inliers = msckf.getInliers();
    const Vector3dArray &msckf_outliers = msckf.getOutliers();
    const size_t rows_msckf = h_msckf.rows();

    /* MSCKF-SLAM */

    // Construct update
    msckf_slam_ = MsckfSlamUpdate(new_msckf_slam_trks_, C_q_G, G_p_C,
                                  triangulator, P, n_poses_max, sigma_img_);

    // Get matrices
    const Matrix &res_msckf_slam = msckf_slam_.getResidual();
    const Matrix &h_msckf_slam = msckf_slam_.getJacobian();
    const Eigen::VectorXd &r_msckf_slam_diag = msckf_slam_.getCovDiag();
    const Vector3dArray &msckf_slam_inliers = msckf_slam_.getInliers();
    const Vector3dArray &msckf_slam_outliers = msckf_slam_.getOutliers();
    const size_t rows_msckf_slam = h_msckf_slam.rows();

    // Stack all MSCKF inliers/outliers
    msckf_inliers_ = msckf_inliers;
    msckf_inliers_.insert(msckf_inliers_.end(), msckf_slam_inliers.begin(),
                          msckf_slam_inliers.end());

    msckf_outliers_ = msckf_outliers;
    msckf_outliers_.insert(msckf_outliers_.end(), msckf_slam_outliers.begin(),
                           msckf_slam_outliers.end());

    // Get SLAM feature priors and inverse-depth pose anchors
    const Matrix &feature_states = state.getFeatureArray();
    const std::vector<int> anchor_idxs = state_manager_.getAnchorIdxs();

    /* SLAM */

    // Contruct update
    slam_ = SlamUpdate(slam_trks_, C_q_G, G_p_C, feature_states, anchor_idxs, P,
                       n_poses_max, sigma_img_);
    const Matrix &res_slam = slam_.getResidual();
    const Matrix &h_slam = slam_.getJacobian();
    const Eigen::VectorXd &r_slam_diag = slam_.getCovDiag();
    const size_t rows_slam = h_slam.rows();

    /* Range-SLAM */

    size_t rows_lrf = 0;
    Matrix h_lrf = Matrix::Zero(0, P.cols()), res_lrf = Matrix::Zero(0, 1);
    Eigen::VectorXd r_lrf_diag;

    if (measurement_.range.timestamp > 0.1 && !slam_trks_.empty()) {
        // 2D image coordinates of the LRF impact point on the ground
        Feature lrf_img_pt;
        lrf_img_pt.setXDist(320.5);
        lrf_img_pt.setYDist(240.5);

        // IDs of the SLAM features in the triangles surrounding the LRF
        const std::vector<int> tr_feat_ids =
                track_manager_.featureTriangleAtPoint(lrf_img_pt, feature_img_);

        // If we found a triangular facet to construct the range update
        if (!tr_feat_ids.empty()) {
            // Contruct update
            const RangeUpdate range_slam(measurement_.range, tr_feat_ids, C_q_G,
                                         G_p_C, feature_states, anchor_idxs, P,
                                         n_poses_max, sigma_range_);
            res_lrf = range_slam.getResidual();
            h_lrf = range_slam.getJacobian();
            r_lrf_diag = range_slam.getCovDiag();
            rows_lrf = 1;

            // Don't reuse measurement
            measurement_.range.timestamp = -1;
        }
    }

    /* Sun Sensor */

    size_t rows_sns = 0;
    Matrix h_sns, res_sns;
    Eigen::VectorXd r_sns_diag;

    if (measurement_.sun_angle.timestamp > -1) {
        // Retrieve body orientation
        const Quaternion &att = state.getOrientation();

        // Construct solar update
        const SolarUpdate solar_update(measurement_.sun_angle, att, P);
        res_sns = solar_update.getResidual();
        h_sns = solar_update.getJacobian();
        r_sns_diag = solar_update.getCovDiag();
        rows_sns = 2;

        // Don't reuse measurement
        measurement_.sun_angle.timestamp = -1;
    }

    /* Combined update */
    const auto rows_total = static_cast<Eigen::Index>(
            rows_msckf + rows_msckf_slam + rows_slam + rows_lrf + rows_sns);
    const auto cols = P.cols();
    h = Matrix::Zero(rows_total, cols);
    Eigen::VectorXd r_diag = Eigen::VectorXd::Ones(rows_total);
    res = Matrix::Zero(rows_total, 1);

    h << h_msckf, h_msckf_slam, h_slam, h_lrf, h_sns;

    r_diag << r_msckf_diag, r_msckf_slam_diag, r_slam_diag, r_lrf_diag,
            r_sns_diag;
    r = r_diag.asDiagonal();

    res << res_msckf, res_msckf_slam, res_slam, res_lrf, res_sns;

    // QR decomposition of the update Jacobian
    applyQRDecomposition(h, res, r);
}

void VioUpdater::postUpdate(State &state, const Matrix &correction) {
    // MSCKF-SLAM feature init
    // Insert all new MSCKF-SLAM features in state and covariance
    if (!new_msckf_slam_trks_.empty()) {
        // TODO(jeff) Do not initialize features which have failed the
        // Mahalanobis test. They need to be removed from the track
        // manager too.
        state_manager_.initMsckfSlamFeatures(state, msckf_slam_.getInitMats(),
                                             correction, sigma_img_);
    }

    // STANDARD SLAM feature initialization
    if (!new_slam_std_trks_.empty()) {
        // Compute inverse-depth coordinates of new SLAM features
        Matrix features_slam_std;
        slam_.computeInverseDepthsNew(new_slam_std_trks_, rho_0_,
                                      features_slam_std);

        // Insert it in state and covariance
        state_manager_.initStandardSlamFeatures(state, features_slam_std,
                                                sigma_img_, sigma_rho_0_);
    }

#ifdef MULTI_UAV
    n_poses_max_ = state.nPosesMax();
#endif
#if defined(MULTI_UAV) && defined(REQUEST_COMM)
    if (frames_min_distance_ > 10) {
      // select keyframe
      Vector3 diff = state.getPosition() - last_pose_;
      auto inverse_depths = state.getFeatureArray();

      double med_depth = 0.0;
      for (int i = 3; i < inverse_depths.rows(); i += 3) {
        if (inverse_depths(i, 0) > 0.001) {  // < 0.001 -> 1000 m too far away
          med_depth += std::abs(1 / inverse_depths(i, 0));
        }
      }
      med_depth /= static_cast<double>(inverse_depths.size()) / 3;

      auto msckf = track_manager_.getMsckfTracks();
      auto slam = track_manager_.normalizeSlamTracks(n_poses_max_);
      auto opp = track_manager_.getOppTracks();
      bool worthy = msckf.size() + slam.size() + opp.size() > 10;
      if ((med_depth > 0.0 && std::abs(diff.norm()) / med_depth > 0.15) &&
          worthy) {  // because the focal length is ~500, dx/med_depth > 0.2
        auto anchors = state_manager_.getAnchorIdxs();
        KeyframePtr kf =
            std::make_shared<Keyframe>(state, anchors, msckf, slam, opp);
        place_recognition_->addKeyframe(kf);
        diff = Vector3(0, 0, 0);

        // used for computing the UAV distance
        last_pose_ = state.getPosition();
        frames_min_distance_ = 0;
      }
      prev_diff_ = diff;
    }
    frames_min_distance_++;
#endif
}

void VioUpdater::applyQRDecomposition(Matrix &h, Matrix &res, Matrix &R) const {
    // Check if the QR decomposition is actually necessary
    unsigned int rowsh(h.rows()), colsh(h.cols());
    bool QR = rowsh > colsh + 1;

    // QR decomposition using Householder transformations (same computational
    // complexity as Givens)
    if (QR) {
        // Compute QR of the augmented [h|res] matrix to avoid forming Q1
        // explicitly (Dongarra et al., Linpack users's guide, 1979)
        Matrix hRes(rowsh, colsh + 1);
        hRes << h, res;
        Eigen::HouseholderQR<Matrix> qr(hRes);
        // Get the upper triangular matrix of the augmented hRes QR
        Matrix Thz = qr.matrixQR().triangularView<Eigen::Upper>();

        // Extract the upper triangular matrix of the h QR
        h = Thz.block(0, 0, colsh, colsh);
        // Extract the projected residual vector
        res = Thz.block(0, colsh, colsh, 1);
        // Form new Kalman update covariance matrix
        const double var_img = sigma_img_ * sigma_img_;
        R = var_img * Matrix::Identity(colsh, colsh);
    }
    // else we leave the inputs unchanged
}
