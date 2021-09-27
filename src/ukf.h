#ifndef UKF_H
#define UKF_H

#include "Eigen/Dense"
#include "measurement_package.h"
#include <pcl/pcl_macros.h>

class UKF {
 public:
  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(MeasurementPackage meas_package);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Prediction(double delta_t);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(MeasurementPackage meas_package);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(MeasurementPackage meas_package);


  // initially set to false. Set to true once initialized the UKF state from either sensor
  bool is_initialized_;

  // if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  // if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  // state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  Eigen::VectorXd x_;

  // state covariance matrix
  Eigen::MatrixXd P_;

  // predicted sigma points matrix
  Eigen::MatrixXd Xsig_pred_;

  // time when the state is true, in us
  long long time_us_;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  // Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  // Laser measurement noise standard deviation position1 in m
  double std_laspx_;

  // Laser measurement noise standard deviation position2 in m
  double std_laspy_;

  // Radar measurement noise standard deviation radius in m
  double std_radr_;

  // Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  // Radar measurement noise standard deviation radius change in m/s
  double std_radrd_ ;

  // Weights of sigma points
  Eigen::VectorXd weights_;

  // State dimension
  int n_x_;

  // Augmented state dimension
  int n_aug_;

  // Number of sigma points
  int n_sig_;

  //sqrt(lambda_ + n_aug_)
  double lambda_plus_aug_;

  // Temp value of predicted augmented state: Dims: [n_aug_, n_sig_]
  Eigen::MatrixXd Xsig_aug_;
  // Sigma point spreading parameter
  double lambda_;


  /////////////////////////////////////////
  //  
  // Functions more-or-less from the exercises
  //
  /////////////////////////////////////////
  inline void AugmentSigmaPoints() {
    Eigen::VectorXd x_aug(n_aug_);
    Eigen::MatrixXd P_aug(n_aug_, n_aug_);
    
    x_aug.setZero();
    x_aug.head(5) = x_;

    P_aug.setZero();
    P_aug.topLeftCorner(5, 5) = P_;
    P_aug(5, 5) = std_a_ * std_a_;
    P_aug(6, 6) = std_yawdd_ * std_yawdd_;

    Eigen::MatrixXd A = P_aug.llt().matrixL();

    // set first column of sigma point matrix
    Xsig_aug_.col(0) = x_aug;

    // set remaining sigma points
    for (int i = 0; i < n_aug_; ++i) {
      Xsig_aug_.col(i + 1) = x_aug + lambda_plus_aug_ * A.col(i);
      Xsig_aug_.col(i + 1 + n_aug_) = x_aug - lambda_plus_aug_ * A.col(i);
    }
  }

  inline void SigmaPointsPrediction(double delta_t) {
    for (int i = 0; i < n_sig_; ++i) {
      double p_x = Xsig_aug_(0, i);
      double p_y = Xsig_aug_(1, i);
      double v = Xsig_aug_(2, i);
      double yaw = Xsig_aug_(3, i);
      double yawd = Xsig_aug_(4, i);
      double nu_a = Xsig_aug_(5, i);
      double nu_yawdd = Xsig_aug_(6, i);

      // predicted state values
      double px_p;
      double py_p;

      // avoid division by zero
      if (std::fabs(yawd) > 0.001) {
        px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
        py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
      }
      else {
        px_p = p_x + v * delta_t * cos(yaw);
        py_p = p_y + v * delta_t * sin(yaw);
      }

      double v_p = v;
      double yaw_p = yaw + yawd * delta_t;
      double yawd_p = yawd;

      // add noise
      px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
      py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
      v_p = v_p + nu_a * delta_t;

      yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
      yawd_p = yawd_p + nu_yawdd * delta_t;

      // write predicted sigma point into right column
      Xsig_pred_(0, i) = px_p;
      Xsig_pred_(1, i) = py_p;
      Xsig_pred_(2, i) = v_p;
      Xsig_pred_(3, i) = yaw_p;
      Xsig_pred_(4, i) = yawd_p;
    }
  }

  inline void ComputeZpredSandResidual(MeasurementPackage meas_package, Eigen::VectorXd& z_pred, Eigen::MatrixXd& Zsig, Eigen::VectorXd& z, Eigen::MatrixXd& R, Eigen::MatrixXd& S, Eigen::VectorXd& z_diff) {

    // z_pred
    z_pred.setZero();
    for (int i = 0; i < n_sig_; ++i) {
      z_pred += weights_(i) * Zsig.col(i);
    }

    // S - covariance
    S.setZero();
    for (int i = 0; i < n_sig_; ++i) {
      Eigen::VectorXd z_diff_loc = Zsig.col(i) - z_pred;
      if (meas_package.is_radar()) {
        z_diff_loc(1) = NormalizeAngle(z_diff_loc(1));
      }
      S += weights_(i) * z_diff_loc * z_diff_loc.transpose();
    }

    S += R;

    // Residual
    z_diff = z - z_pred;

  }

  inline double NormalizeAngle(double angle) {
    while (angle > M_PI) {
      angle -= 2.0 * M_PI;
    }
    while (angle < -M_PI) {
      angle += 2.0 * M_PI;
    }
    return angle;
  }

  inline void MeasurementUpdate(MeasurementPackage& measurement_package, Eigen::MatrixXd& Zsig, int n_z, Eigen::MatrixXd& S, Eigen::VectorXd& z_pred, Eigen::VectorXd& z_diff) {

    Eigen::MatrixXd Tc(n_x_, n_z);
    Tc.fill(0.0);
    for (int i = 0; i < n_sig_; ++i) {
      // residual
      Eigen::VectorXd z_diff = Zsig.col(i) - z_pred;

      if (measurement_package.is_radar()) {
        z_diff(1) = NormalizeAngle(z_diff(1));
      }

      Eigen::VectorXd x_diff = Xsig_pred_.col(i) - x_;
      x_diff(3) = NormalizeAngle(x_diff(3));

      Tc += weights_(i) * x_diff * z_diff.transpose();
    }

    // Kalman gain K;
    Eigen::MatrixXd Si = S.inverse();
    Eigen::MatrixXd K = Tc * Si;

    // Angle normalization
    if (measurement_package.is_radar()) {
      z_diff(1) = NormalizeAngle(z_diff(1));
    }

    x_ += K * z_diff;
    P_ -= K * S * K.transpose();
  }

  inline void PredictMeanAndCovariance() {
    Eigen::VectorXd x(n_x_);
    
    x.setZero();
    
    for (int i = 0; i < n_sig_; ++i) {
      x += weights_(i) * Xsig_pred_.col(i);
    }

    Eigen::MatrixXd P(n_x_, n_x_);

    P.setZero();

    Eigen::VectorXd x_diff;
    for (int i = 0; i < n_sig_; ++i) {  // iterate over sigma points
      x_diff = Xsig_pred_.col(i) - x;
      x_diff(3) = NormalizeAngle(x_diff(3));

      P += weights_(i) * x_diff * x_diff.transpose();
    }

    x_ = x;
    P_ = P;
  }

};

#endif  // UKF_H