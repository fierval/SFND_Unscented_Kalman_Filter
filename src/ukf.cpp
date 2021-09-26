#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() 
  : nis_(2) {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */

  n_aug_ = 7;
  n_x_ = 5;

  n_sig_ = 2 * n_aug_ + 1;

  Xsig_pred_.resize(n_x_, n_sig_);
  Xsig_aug_.resize(n_aug_, n_sig_);
  weights_.resize(n_sig_);

  weights_.fill(0.5 / (lambda_ + n_aug_));
  weights_(0) = lambda_ / (lambda_ + n_aug_);

  is_initialized_ = -1;
  lambda_plus_aug_ = std::sqrt(lambda_ + n_aug_);

  // pre-init useful vars
  P_.setIdentity();
  x_.setZero();
  std::fill_n(nis_.begin(), 2, 0);
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  
  // Initialization is delegated to each sensor.
  // Makes for cleaner code
  switch (meas_package.sensor_type_)
  {
  case MeasurementPackage::SensorType::LASER:
  {
    UpdateLidar(meas_package);
    break;
  }
  case MeasurementPackage::SensorType::RADAR:
  {
    UpdateRadar(meas_package);
    break;
  }
  default:
    // any sensor we don't know about?
    assert(false);
    break;
  }

  time_us_ = meas_package.timestamp_;
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
  
  AugmentedSigmaPoints();
  SigmaPointsPrediction(delta_t);
  PredictMeanAndCovariance();

}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
  if (!is_initialized_) {
    x_.head(2) << meas_package.raw_measurements_.head(2);
    is_initialized_ = true;
  }

  double delta_t = static_cast<double>(meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;
  Prediction(delta_t);

  int n_z = 2;
  Eigen::MatrixXd S(n_z, n_z);  // Measurement covariance matrix (n_z = 2 for lidar, n_z = 3 for radar)

  // create matrix for sigma points in measurement space
  Eigen::MatrixXd Zsig(n_z, n_sig_);

  // transform sigma points into measurement space
  Zsig.row(0) = Xsig_pred_.row(0);
  Zsig.row(1) = Xsig_pred_.row(1);

  Eigen::MatrixXd R(n_z, n_z);
  Eigen::VectorXd z(n_z);

  z << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1];
  R << std_laspx_ * std_laspx_, 0, 0, std_laspy_* std_laspy_;

  Eigen::VectorXd z_pred(n_z);  // Mean predicted measurement
  Eigen::VectorXd z_diff(n_z);
  ComputeZpredSandResidual(meas_package, z_pred, Zsig, z, R, S, z_diff);

  // Update!
  MeasurementUpdate(meas_package, Zsig, n_z, S, z_pred, z_diff);
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   * 
   */
  if (!is_initialized_) {

    double r = meas_package.raw_measurements_[0];
    double phi = meas_package.raw_measurements_[1];
    double r_dot = meas_package.raw_measurements_[2];
    double cos_phi = std::cos(phi);
    double sin_phi = std::sin(phi);

    x_ <<
      r * cos_phi,
      r* sin_phi,
      r_dot; // not quite!

    is_initialized_ = true;
  }

  double delta_t = static_cast<double>(meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;
  Prediction(delta_t);

  int n_z = 3;
  Eigen::MatrixXd S(n_z, n_z);  // Measurement covariance matrix (n_z = 2 for lidar, n_z = 3 for radar)

  // transform sigma points into measurement space
  Eigen::MatrixXd Zsig(n_z, n_sig_);

  for (int i = 0; i < n_sig_; ++i) {
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    double v1 = v * cos(yaw);
    double v2 = v * sin(yaw);

    Zsig(0, i) = std::sqrt(p_x * p_x + p_y * p_y);                          // r
    Zsig(1, i) = std::atan2(p_y, p_x);                                      // phi
    Zsig(2, i) = (p_x * v1 + p_y * v2) / std::sqrt(p_x * p_x + p_y * p_y);  // r_dot
  }

  Eigen::MatrixXd R(n_z, n_z);
  Eigen::VectorXd z(n_z);

  z << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], meas_package.raw_measurements_[2];
  R << std_radr_ * std_radr_, 0, 0, 0, std_radphi_* std_radphi_, 0, 0, 0, std_radrd_* std_radrd_;

  Eigen::VectorXd z_pred(n_z);  // Mean predicted measurement
  Eigen::VectorXd z_diff(n_z);
  ComputeZpredSandResidual(meas_package, z_pred, Zsig, z, R, S, z_diff);

  // Update!
  MeasurementUpdate(meas_package, Zsig, n_z, S, z_pred, z_diff);
}