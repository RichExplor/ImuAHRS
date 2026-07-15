#ifndef MADGWICK_AHRS_H
#define MADGWICK_AHRS_H

#include <Eigen/Dense>
#include "ImuData.h"
#include "QuaternionUtils.h"

class MadgwickAHRS {
public:
    struct Config {
        double beta;
        bool use_magnetometer;
        bool adaptive_beta;
        double accel_threshold;
        
        Config() 
            : beta(0.05), use_magnetometer(false), 
              adaptive_beta(true), accel_threshold(0.2) {}
    };
    
    struct State {
        Eigen::Quaterniond q;
        
        State() : q(1,0,0,0) {}
    };

    MadgwickAHRS(const Config& config = Config());
    
    void init(const Eigen::Quaterniond& q_init);
    
    void update(const ImuData& imu, double dt);
    
    void update(const Eigen::Vector3d& accel, const Eigen::Vector3d& gyro, 
                const Eigen::Vector3d& mag, double dt);
    
    const Eigen::Quaterniond& getOrientation() const { return state_.q; }
    
    Eigen::Vector3d getEulerAngles() const;
    
    Eigen::Matrix3d getRotationMatrix() const;
    
    void reset();
    
    void setConfig(const Config& config) { config_ = config; }
    const Config& getConfig() const { return config_; }
    
private:
    Config config_;
    State state_;
    
    static void normalizeVector(Eigen::Vector3d& v);
    
    double computeAdaptiveBeta(const Eigen::Vector3d& accel) const;
    
    Eigen::Vector4d gradientStep6DOF(const Eigen::Vector4d& q, const Eigen::Vector3d& accel) const;
    
    Eigen::Vector4d gradientStep9DOF(const Eigen::Vector4d& q, const Eigen::Vector3d& accel, const Eigen::Vector3d& mag) const;
};

#endif // MADGWICK_AHRS_H
