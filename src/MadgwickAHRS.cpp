#include "MadgwickAHRS.h"
#include "QuaternionUtils.h"
#include <cmath>
#include <algorithm>

using namespace QuaternionUtils;

MadgwickAHRS::MadgwickAHRS(const Config& config) : config_(config), state_() {}

void MadgwickAHRS::init(const Eigen::Quaterniond& q_init) {
    state_.q = q_init.normalized();
}

void MadgwickAHRS::update(const ImuData& imu, double dt) {
    update(imu.accel, imu.gyro, imu.mag, dt);
}

void MadgwickAHRS::update(const Eigen::Vector3d& accel, const Eigen::Vector3d& gyro,
                          const Eigen::Vector3d& mag, double dt) {
    if (dt <= 0) return;

    // Validate accelerometer (check before normalization)
    double accel_norm = accel.norm();
    if (accel_norm < 1e-6) return;

    // Normalize accelerometer
    Eigen::Vector3d a = accel / accel_norm;

    // Normalize magnetometer if used
    Eigen::Vector3d m = mag;
    double mag_norm = m.norm();
    bool use_mag = config_.use_magnetometer && mag_norm > 1e-6;
    if (use_mag) {
        m /= mag_norm;
    }

    // Compute adaptive beta: trust accel LESS when accel deviates from 1g
    double beta = config_.beta;
    if (config_.adaptive_beta) {
        // Reduce trust in accel (higher beta effectively, but we reduce beta to slow correction)
        beta = computeAdaptiveBeta(accel);
    }

    // Convert quaternion to Vector4d for computation (q = [w, x, y, z])
    Eigen::Vector4d q(state_.q.w(), state_.q.x(), state_.q.y(), state_.q.z());

    // Gradient descent step
    Eigen::Vector4d grad;
    if (use_mag) {
        grad = gradientStep9DOF(q, a, m);
    } else {
        grad = gradientStep6DOF(q, a);
    }

    // Compute rate of change of quaternion from gyroscope
    double qw = state_.q.w(), qx = state_.q.x(), qy = state_.q.y(), qz = state_.q.z();
    double wx = gyro.x(), wy = gyro.y(), wz = gyro.z();

    Eigen::Vector4d q_dot_gyro;
    q_dot_gyro(0) = -0.5 * (qx*wx + qy*wy + qz*wz);
    q_dot_gyro(1) =  0.5 * (qw*wx + qy*wz - qz*wy);
    q_dot_gyro(2) =  0.5 * (qw*wy - qx*wz + qz*wx);
    q_dot_gyro(3) =  0.5 * (qw*wz + qx*wy - qy*wx);

    // Apply gradient descent correction
    double g_norm = grad.norm();
    if (g_norm > 1e-6) {
        grad /= g_norm;
    }

    Eigen::Vector4d q_dot = q_dot_gyro - beta * grad;

    // Integrate
    qw += q_dot(0) * dt;
    qx += q_dot(1) * dt;
    qy += q_dot(2) * dt;
    qz += q_dot(3) * dt;

    // Normalize
    double norm = std::sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
    if (norm > 1e-6) {
        qw /= norm; qx /= norm; qy /= norm; qz /= norm;
    }

    state_.q = Eigen::Quaterniond(qw, qx, qy, qz);
}

double MadgwickAHRS::computeAdaptiveBeta(const Eigen::Vector3d& accel) const {
    double accel_mag = accel.norm();
    double diff = std::abs(accel_mag - 9.81);
    if (diff > config_.accel_threshold) {
        // Reduce beta (reduce correction strength) when accel deviates from 1g
        // Because accel reading is contaminated by linear acceleration
        return config_.beta * (config_.accel_threshold / (diff + 1e-6));
    }
    return config_.beta;
}

void MadgwickAHRS::normalizeVector(Eigen::Vector3d& v) {
    double norm = v.norm();
    if (norm > 1e-6) {
        v /= norm;
    }
}

Eigen::Vector4d MadgwickAHRS::gradientStep6DOF(const Eigen::Vector4d& q, const Eigen::Vector3d& accel) const {

    double qw = q(0), qx = q(1), qy = q(2), qz = q(3);
    double ax = accel.x(), ay = accel.y(), az = accel.z();

    Eigen::Vector3d f;
    f(0) = 2.0 * (qx*qz - qw*qy) - ax;
    f(1) = 2.0 * (qw*qx + qy*qz) - ay;
    f(2) = 2.0 * (0.5 - qx*qx - qy*qy) - az;

    // Jacobian of f w.r.t. q
    Eigen::Matrix<double, 3, 4> J;
    J(0,0) = -2.0*qy;  J(0,1) =  2.0*qz;  J(0,2) = -2.0*qw;  J(0,3) =  2.0*qx;
    J(1,0) =  2.0*qx;  J(1,1) =  2.0*qw;  J(1,2) =  2.0*qz;  J(1,3) =  2.0*qy;
    J(2,0) =  0.0;     J(2,1) = -4.0*qx;  J(2,2) = -4.0*qy;  J(2,3) =  0.0;

    Eigen::Vector4d grad = J.transpose() * f;

    return grad;
}

Eigen::Vector4d MadgwickAHRS::gradientStep9DOF(const Eigen::Vector4d& q, const Eigen::Vector3d& accel, const Eigen::Vector3d& mag) const {
    double qw = q(0), qx = q(1), qy = q(2), qz = q(3);
    double mx = mag.x(), my = mag.y(), mz = mag.z();

    // 6-DOF gradient
    Eigen::Vector4d grad_acc = gradientStep6DOF(q, accel);

    // Magnetometer: reference direction in world is [bx, 0, bz] after tilt compensation
    double hx = 2.0*mx*(0.5 - qy*qy - qz*qz) + 2.0*my*(qx*qy - qw*qz) + 2.0*mz*(qx*qz + qw*qy);
    double hy = 2.0*mx*(qx*qy + qw*qz) + 2.0*my*(0.5 - qx*qx - qz*qz) + 2.0*mz*(qy*qz - qw*qx);
    double hz = 2.0*mx*(qx*qz - qw*qy) + 2.0*my*(qy*qz + qw*qx) + 2.0*mz*(0.5 - qx*qx - qy*qy);
    double bx = std::sqrt(hx*hx + hy*hy);
    double bz = hz;

    // Objective function for magnetometer
    Eigen::Vector3d f_mag;
    f_mag(0) = 2.0*bx*(0.5 - qy*qy - qz*qz) + 2.0*bz*(qx*qz - qw*qy) - mx;
    f_mag(1) = 2.0*bx*(qx*qy - qw*qz) + 2.0*bz*(qw*qx + qy*qz) - my;
    f_mag(2) = 2.0*bx*(qw*qy + qx*qz) + 2.0*bz*(0.5 - qx*qx - qy*qy) - mz;

    // Jacobian of f_mag w.r.t. q
    Eigen::Matrix<double, 3, 4> J_mag;
    J_mag(0,0) = -2.0*bz*qy;
    J_mag(0,1) =  2.0*bz*qz;
    J_mag(0,2) = -4.0*bx*qy - 2.0*bz*qw;
    J_mag(0,3) = -4.0*bx*qz + 2.0*bz*qx;

    // Row 1: f_mag(1) = bx*(2*qx*qy - 2*qw*qz) + bz*(2*qw*qx + 2*qy*qz) - my
    J_mag(1,0) = -2.0*bx*qz + 2.0*bz*qx;
    J_mag(1,1) =  2.0*bx*qy + 2.0*bz*qw;
    J_mag(1,2) =  2.0*bx*qx + 2.0*bz*qz;
    J_mag(1,3) = -2.0*bx*qw + 2.0*bz*qy;

    // Row 2: f_mag(2) = bx*(2*qw*qy + 2*qx*qz) + bz*(1 - 2*qx^2 - 2*qy^2) - mz
    J_mag(2,0) =  2.0*bx*qy;
    J_mag(2,1) =  2.0*bx*qz - 4.0*bz*qx;
    J_mag(2,2) =  2.0*bx*qw - 4.0*bz*qy;
    J_mag(2,3) =  2.0*bx*qx;

    Eigen::Vector4d grad_mag = J_mag.transpose() * f_mag;

    // Combine gradients
    Eigen::Vector4d grad = grad_acc + grad_mag;

    return grad;
}

Eigen::Vector3d MadgwickAHRS::getEulerAngles() const {
    return toEulerZYX(state_.q);
}

Eigen::Matrix3d MadgwickAHRS::getRotationMatrix() const {
    return toRotationMatrix(state_.q);
}

void MadgwickAHRS::reset() {
    state_.q.setIdentity();
}
