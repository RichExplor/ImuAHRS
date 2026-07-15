#include "ImuData.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <random>
#include <cmath>

// Coordinate frame used throughout this project:
//   X = Magnetic North, Y = West, Z = Up  (right-handed: North x West = Up).
// Gravity acceleration in world frame: [0, 0, -g].
// Accelerometer measures specific force = -gravity + motion accel,
// therefore at rest the accelerometer reads [0, 0, +g].

ImuDataset ImuDataset::loadFromCsv(const std::string& filename, double dt) {
    ImuDataset dataset;
    dataset.dt = dt;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return dataset;
    }

    std::string line;
    bool first_line = true;
    while (std::getline(file, line)) {
        if (first_line) {
            // Skip header
            first_line = false;
            continue;
        }

        std::stringstream ss(line);
        std::string token;
        std::vector<double> values;

        while (std::getline(ss, token, ',')) {
            try {
                values.push_back(std::stod(token));
            } catch (...) {
                continue;
            }
        }

        if (values.size() >= 10) {
            ImuData data;
            data.timestamp = values[0];
            data.accel = Eigen::Vector3d(values[1], values[2], values[3]);
            data.gyro = Eigen::Vector3d(values[4], values[5], values[6]);
            data.mag = Eigen::Vector3d(values[7], values[8], values[9]);
            if (values.size() > 10) data.temperature = values[10];
            dataset.data.push_back(data);
        }
    }

    return dataset;
}

ImuDataset ImuDataset::generateSynthetic(double duration, double dt, double noise_level) {
    ImuDataset dataset;
    dataset.dt = dt;

    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::normal_distribution<double> accel_noise(0, noise_level);
    std::normal_distribution<double> gyro_noise(0, noise_level * 0.01);
    std::normal_distribution<double> mag_noise(0, noise_level * 10);

    int num_samples = static_cast<int>(duration / dt);
    dataset.data.reserve(num_samples);

    // Simulate a trajectory with time-varying yaw/pitch/roll.
    for (int i = 0; i < num_samples; ++i) {
        double t = i * dt;

        // Ground truth orientation: rotating around Z with oscillation in pitch/roll.
        double yaw   = 0.5 * t;
        double pitch = 0.2 * std::sin(0.5 * t);
        double roll  = 0.1 * std::cos(0.7 * t);

        // Convert ZYX Euler angles (yaw, pitch, roll) to a world-to-body quaternion.
        // This quaternion satisfies v_body = q * v_world for any world vector v_world.
        Eigen::Quaterniond q_true =
            Eigen::Quaterniond(Eigen::AngleAxisd(yaw,   Eigen::Vector3d::UnitZ())) *
            Eigen::Quaterniond(Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY())) *
            Eigen::Quaterniond(Eigen::AngleAxisd(roll,  Eigen::Vector3d::UnitX()));

        // True angular velocity in body frame, derived from ZYX Euler rates.
        // For ZYX Euler (psi=yaw, theta=pitch, phi=roll):
        //   omega_x =  phi_dot - psi_dot * sin(theta)
        //   omega_y =  theta_dot * cos(phi) + psi_dot * sin(phi) * cos(theta)
        //   omega_z = -theta_dot * sin(phi) + psi_dot * cos(phi) * cos(theta)
        double yaw_dot   = 0.5;
        double pitch_dot = 0.2 * 0.5 * std::cos(0.5 * t);
        double roll_dot  = -0.1 * 0.7 * std::sin(0.7 * t);

        Eigen::Vector3d omega_true;
        omega_true.x() = roll_dot - yaw_dot * std::sin(pitch);
        omega_true.y() = pitch_dot * std::cos(roll) + yaw_dot * std::sin(roll) * std::cos(pitch);
        omega_true.z() = -pitch_dot * std::sin(roll) + yaw_dot * std::cos(roll) * std::cos(pitch);

        // True specific force in world frame (-gravity at rest, since Z is Up).
        Eigen::Vector3d a_world(0.0, 0.0, 9.81);

        // Transform to body frame: a_body = q^* * a_world.
        Eigen::Vector3d a_body = q_true.conjugate() * a_world;

        // True magnetic field in world frame (Magnetic North = X, magnitude = 50 uT).
        // Zero magnetic dip keeps the 9-DOF test clean.
        Eigen::Vector3d m_world(50.0, 0.0, 0.0);
        Eigen::Vector3d m_body = q_true.conjugate() * m_world;

        // Add noise
        ImuData data;
        data.timestamp = t;
        data.accel = a_body + Eigen::Vector3d(accel_noise(gen), accel_noise(gen), accel_noise(gen));
        data.gyro = omega_true + Eigen::Vector3d(gyro_noise(gen), gyro_noise(gen), gyro_noise(gen));
        data.mag = m_body + Eigen::Vector3d(mag_noise(gen), mag_noise(gen), mag_noise(gen));

        dataset.data.push_back(data);
    }

    return dataset;
}

GroundTruthDataset GroundTruthDataset::generateSynthetic(double duration, double dt) {
    GroundTruthDataset dataset;
    dataset.data.reserve(static_cast<size_t>(duration / dt));

    for (double t = 0; t < duration; t += dt) {
        double yaw   = 0.5 * t;
        double pitch = 0.2 * std::sin(0.5 * t);
        double roll  = 0.1 * std::cos(0.7 * t);

        Eigen::Quaterniond q_true =
            Eigen::Quaterniond(Eigen::AngleAxisd(yaw,   Eigen::Vector3d::UnitZ())) *
            Eigen::Quaterniond(Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY())) *
            Eigen::Quaterniond(Eigen::AngleAxisd(roll,  Eigen::Vector3d::UnitX()));

        dataset.data.emplace_back(t, q_true);
    }

    return dataset;
}
