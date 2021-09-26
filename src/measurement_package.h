#ifndef MEASUREMENT_PACKAGE_H_
#define MEASUREMENT_PACKAGE_H_

#include "Eigen/Dense"

class MeasurementPackage {
public:
  long timestamp_;
  inline bool is_radar() { return sensor_type_ == SensorType::RADAR; }

  enum SensorType : int{
    LASER = 0,
    RADAR = 1
  } sensor_type_;

  Eigen::VectorXd raw_measurements_;

};

#endif /* MEASUREMENT_PACKAGE_H_ */