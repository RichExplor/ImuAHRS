#ifndef IMU_DATA_H
#define IMU_DATA_H

#include <Eigen/Dense>
#include <vector>
#include <string>

struct ImuData {
    double timestamp;        // Timestamp in seconds
    Eigen::Vector3d accel;   // Accelerometer (m/s^2), body frame
    Eigen::Vector3d gyro;    // Gyroscope (rad/s), body frame
    Eigen::Vector3d mag;     // Magnetometer (uT or normalized), body frame
    double temperature;      // Temperature (deg C)
    
    ImuData() : timestamp(0), accel(0,0,0), gyro(0,0,0), mag(0,0,0), temperature(25) {}
    
    ImuData(double ts, const Eigen::Vector3d& a, const Eigen::Vector3d& g, 
            const Eigen::Vector3d& m, double temp = 25)
        : timestamp(ts), accel(a), gyro(g), mag(m), temperature(temp) {}
    
    // Check if data is valid
    bool isValid() const {
        return timestamp > 0 && 
               accel.norm() > 0 && 
               gyro.norm() >= 0 &&
               std::isfinite(timestamp) &&
               std::isfinite(accel.x()) && std::isfinite(accel.y()) && std::isfinite(accel.z()) &&
               std::isfinite(gyro.x()) && std::isfinite(gyro.y()) && std::isfinite(gyro.z());
    }
};

// IMU sensor configuration
struct ImuConfig {
    Eigen::Vector3d accel_bias;    // Accelerometer bias (m/s^2)
    Eigen::Vector3d gyro_bias;     // Gyroscope bias (rad/s)
    Eigen::Vector3d mag_bias;      // Magnetometer bias (uT)
    Eigen::Matrix3d accel_scale;   // Accelerometer scale/misalignment matrix
    Eigen::Matrix3d gyro_scale;    // Gyroscope scale/misalignment matrix
    Eigen::Matrix3d mag_scale;     // Magnetometer scale/misalignment matrix
    Eigen::Vector3d mag_hard_iron; // Hard iron offset
    Eigen::Matrix3d mag_soft_iron; // Soft iron matrix
    
    ImuConfig() 
        : accel_bias(0,0,0), gyro_bias(0,0,0), mag_bias(0,0,0),
          accel_scale(Eigen::Matrix3d::Identity()),
          gyro_scale(Eigen::Matrix3d::Identity()),
          mag_scale(Eigen::Matrix3d::Identity()),
          mag_hard_iron(0,0,0),
          mag_soft_iron(Eigen::Matrix3d::Identity()) {}
};

// Apply calibration to raw IMU data
inline ImuData calibrate(const ImuData& raw, const ImuConfig& config) {
    ImuData calibrated = raw;
    calibrated.accel = config.accel_scale * (raw.accel - config.accel_bias);
    calibrated.gyro = config.gyro_scale * (raw.gyro - config.gyro_bias);
    
    // Magnetometer calibration: hard iron + soft iron
    Eigen::Vector3d mag_corrected = raw.mag - config.mag_hard_iron;
    calibrated.mag = config.mag_scale * config.mag_soft_iron * mag_corrected;
    
    return calibrated;
}

// Dataset for IMU data
struct ImuDataset {
    std::vector<ImuData> data;
    double dt;  // time step (assumed constant)
    
    ImuDataset() : dt(0.01) {}
    
    size_t size() const { return data.size(); }
    const ImuData& operator[](size_t i) const { return data[i]; }
    ImuData& operator[](size_t i) { return data[i]; }
    
    // Load from CSV file (timestamp, ax, ay, az, gx, gy, gz, mx, my, mz)
    static ImuDataset loadFromCsv(const std::string& filename, double dt = 0.01);
    
    // Generate synthetic test data (simulated IMU with known ground truth)
    static ImuDataset generateSynthetic(double duration = 10.0, double dt = 0.01, double noise_level = 0.01);
};

struct GroundTruth {
    double timestamp;
    Eigen::Quaterniond q;  // world to body quaternion
    
    GroundTruth() : timestamp(0), q(1,0,0,0) {}
    GroundTruth(double ts, const Eigen::Quaterniond& q_) : timestamp(ts), q(q_) {}
};

struct GroundTruthDataset {
    std::vector<GroundTruth> data;
    
    size_t size() const { return data.size(); }
    const GroundTruth& operator[](size_t i) const { return data[i]; }
    GroundTruth& operator[](size_t i) { return data[i]; }
    
    // Generate ground truth for synthetic data
    static GroundTruthDataset generateSynthetic(double duration = 10.0, double dt = 0.01);
};

#endif // IMU_DATA_H
