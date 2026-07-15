#ifndef QUATERNION_UTILS_H
#define QUATERNION_UTILS_H

#include <Eigen/Dense>
#include <cmath>

// Quaternion utilities for attitude representation
namespace QuaternionUtils {
    // Normalize quaternion
    inline Eigen::Quaterniond normalize(const Eigen::Quaterniond& q) {
        Eigen::Quaterniond qn = q;
        qn.normalize();
        return qn;
    }
    
    // Quaternion multiplication: q1 * q2
    inline Eigen::Quaterniond multiply(const Eigen::Quaterniond& q1, const Eigen::Quaterniond& q2) {
        return q1 * q2;
    }
    
    // Quaternion from angular velocity (integrate)
    // q_dot = 0.5 * q * [0, wx, wy, wz]
    inline Eigen::Quaterniond integrate(const Eigen::Quaterniond& q, const Eigen::Vector3d& omega, double dt) {
        if (dt <= 0 || omega.norm() < 1e-10) return q;
        
        // q_dot = 0.5 * q * [0, omega]
        Eigen::Quaterniond q_dot(0, 
            0.5 * (q.w() * omega.x() + q.y() * omega.z() - q.z() * omega.y()),
            0.5 * (q.w() * omega.y() - q.x() * omega.z() + q.z() * omega.x()),
            0.5 * (q.w() * omega.z() + q.x() * omega.y() - q.y() * omega.x())
        );
        
        Eigen::Quaterniond q_new;
        q_new.w() = q.w() + q_dot.w() * dt;
        q_new.x() = q.x() + q_dot.x() * dt;
        q_new.y() = q.y() + q_dot.y() * dt;
        q_new.z() = q.z() + q_dot.z() * dt;
        return normalize(q_new);
    }
    
    // Convert quaternion to Euler angles (roll, pitch, yaw) in radians
    // ZYX convention (yaw-pitch-roll)
    inline Eigen::Vector3d toEulerZYX(const Eigen::Quaterniond& q) {
        Eigen::Vector3d euler;
        double sinr_cosp = 2.0 * (q.w() * q.x() + q.y() * q.z());
        double cosr_cosp = 1.0 - 2.0 * (q.x() * q.x() + q.y() * q.y());
        euler.x() = std::atan2(sinr_cosp, cosr_cosp);
        
        double sinp = 2.0 * (q.w() * q.y() - q.z() * q.x());
        if (std::abs(sinp) >= 1.0)
            euler.y() = std::copysign(M_PI / 2.0, sinp);
        else
            euler.y() = std::asin(sinp);
        
        double siny_cosp = 2.0 * (q.w() * q.z() + q.x() * q.y());
        double cosy_cosp = 1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z());
        euler.z() = std::atan2(siny_cosp, cosy_cosp);
        
        return euler;
    }
    
    // Convert Euler angles to quaternion (ZYX)
    inline Eigen::Quaterniond fromEulerZYX(const Eigen::Vector3d& euler) {
        double roll = euler.x();
        double pitch = euler.y();
        double yaw = euler.z();
        
        Eigen::Quaterniond q_z(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()));
        Eigen::Quaterniond q_y(Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()));
        Eigen::Quaterniond q_x(Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()));
        
        return q_z * q_y * q_x;
    }
    
    // Convert quaternion to rotation matrix (body to world)
    inline Eigen::Matrix3d toRotationMatrix(const Eigen::Quaterniond& q) {
        return q.toRotationMatrix();
    }
    
    // Rotate vector from body frame to world frame
    inline Eigen::Vector3d rotateBodyToWorld(const Eigen::Quaterniond& q, const Eigen::Vector3d& v_body) {
        return q * v_body;
    }
    
    // Rotate vector from world frame to body frame
    inline Eigen::Vector3d rotateWorldToBody(const Eigen::Quaterniond& q, const Eigen::Vector3d& v_world) {
        return q.conjugate() * v_world;
    }
    
    // Compute quaternion error between estimated and true orientation
    // Returns angle-axis representation of error (3D vector, angle = norm)
    inline Eigen::Vector3d orientationError(const Eigen::Quaterniond& q_est, const Eigen::Quaterniond& q_true) {
        Eigen::Quaterniond q_err = q_est.conjugate() * q_true;
        if (q_err.w() < 0) q_err.coeffs() *= -1;
        double angle = 2.0 * std::acos(std::min(1.0, std::max(-1.0, q_err.w())));
        if (angle < 1e-10) return Eigen::Vector3d::Zero();
        Eigen::Vector3d axis = q_err.vec();
        axis.normalize();
        return axis * angle;
    }
    
    // Compute angular error in degrees
    inline double orientationErrorDeg(const Eigen::Quaterniond& q_est, const Eigen::Quaterniond& q_true) {
        return orientationError(q_est, q_true).norm() * 180.0 / M_PI;
    }
    
    // Slerp between two quaternions
    inline Eigen::Quaterniond slerp(const Eigen::Quaterniond& q1, const Eigen::Quaterniond& q2, double t) {
        return q1.slerp(t, q2);
    }
    
    // Quaternion conjugate
    inline Eigen::Quaterniond conjugate(const Eigen::Quaterniond& q) {
        return q.conjugate();
    }
    
    // Quaternion inverse (same as conjugate for unit quaternions)
    inline Eigen::Quaterniond inverse(const Eigen::Quaterniond& q) {
        return q.inverse();
    }
}

#endif // QUATERNION_UTILS_H
