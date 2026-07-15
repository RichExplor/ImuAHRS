#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <chrono>
#include "ImuData.h"
#include "MadgwickAHRS.h"
#include "QuaternionUtils.h"

using namespace QuaternionUtils;

struct TestResult {
    std::string algorithm_name;
    double rmse_roll_deg;
    double rmse_pitch_deg;
    double rmse_yaw_deg;
    double rmse_orientation_deg;
    double max_error_deg;
    double convergence_time_s;
    double avg_update_time_us;
};

void printSeparator() {
    std::cout << std::string(80, '=') << std::endl;
}

void printResult(const TestResult& result) {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Algorithm: " << result.algorithm_name << std::endl;
    std::cout << "  RMSE Roll:     " << std::setw(8) << result.rmse_roll_deg << " deg" << std::endl;
    std::cout << "  RMSE Pitch:    " << std::setw(8) << result.rmse_pitch_deg << " deg" << std::endl;
    std::cout << "  RMSE Yaw:      " << std::setw(8) << result.rmse_yaw_deg << " deg" << std::endl;
    std::cout << "  RMSE Orientation: " << std::setw(8) << result.rmse_orientation_deg << " deg" << std::endl;
    std::cout << "  Max Error:     " << std::setw(8) << result.max_error_deg << " deg" << std::endl;
    std::cout << "  Convergence:   " << std::setw(8) << result.convergence_time_s << " s" << std::endl;
    std::cout << "  Avg Update:    " << std::setw(8) << result.avg_update_time_us << " us" << std::endl;
}

TestResult runTest(const std::string& name, const ImuDataset& dataset, 
                   const GroundTruthDataset& ground_truth, 
                   bool use_magnetometer, double beta = 0.01) {
    TestResult result;
    result.algorithm_name = name;
    if (name.find("Madgwick") != std::string::npos) {
        MadgwickAHRS::Config config;
        config.beta = beta;
        config.use_magnetometer = use_magnetometer;
        config.adaptive_beta = true;
        config.accel_threshold = 0.2;
        
        MadgwickAHRS ahrs(config);
        ahrs.init(ground_truth[0].q);
        
        double total_update_time = 0;
        bool converged = false;
        double convergence_time = 0;
        double sum_sq_roll = 0, sum_sq_pitch = 0, sum_sq_yaw = 0, sum_sq_orient = 0;
        double max_error = 0;
        int converged_count = 0;
        
        for (size_t i = 0; i < dataset.size(); ++i) {
            const auto& imu = dataset[i];
            const auto& gt = ground_truth[i];
            
            auto start = std::chrono::high_resolution_clock::now();
            ahrs.update(imu, dataset.dt);
            auto end = std::chrono::high_resolution_clock::now();
            total_update_time += std::chrono::duration<double, std::micro>(end - start).count();
            
            Eigen::Quaterniond q_est = ahrs.getOrientation();
            double error = orientationErrorDeg(q_est, gt.q);
            Eigen::Vector3d euler_est = toEulerZYX(q_est);
            Eigen::Vector3d euler_gt = toEulerZYX(gt.q);
            
            double err_roll = std::abs(euler_est.x() - euler_gt.x()) * 180.0 / M_PI;
            double err_pitch = std::abs(euler_est.y() - euler_gt.y()) * 180.0 / M_PI;
            double err_yaw = std::abs(euler_est.z() - euler_gt.z()) * 180.0 / M_PI;
            
            if (err_yaw > 180) err_yaw = 360 - err_yaw;
            
            sum_sq_roll += err_roll * err_roll;
            sum_sq_pitch += err_pitch * err_pitch;
            sum_sq_yaw += err_yaw * err_yaw;
            sum_sq_orient += error * error;
            max_error = std::max(max_error, error);
            
            if (error < 5.0) {
                converged_count++;
                if (!converged && converged_count >= 100) {
                    converged = true;
                    convergence_time = imu.timestamp;
                }
            } else {
                converged_count = 0;
            }
        }
        
        int n = dataset.size();
        result.rmse_roll_deg = std::sqrt(sum_sq_roll / n);
        result.rmse_pitch_deg = std::sqrt(sum_sq_pitch / n);
        result.rmse_yaw_deg = std::sqrt(sum_sq_yaw / n);
        result.rmse_orientation_deg = std::sqrt(sum_sq_orient / n);
        result.max_error_deg = max_error;
        result.convergence_time_s = convergence_time;
        result.avg_update_time_us = total_update_time / n;
    }
    
    return result;
}

int main() {
    std::cout << "IMU AHRS Algorithm Comparison: Mahony vs Madgwick" << std::endl;
    printSeparator();
    
    // Generate synthetic test data
    double duration = 20.0;  // 20 seconds
    double dt = 0.01;        // 100 Hz
    double noise_level = 0.02; // 2% noise
    
    std::cout << "Generating synthetic test data..." << std::endl;
    ImuDataset dataset = ImuDataset::generateSynthetic(duration, dt, noise_level);
    GroundTruthDataset ground_truth = GroundTruthDataset::generateSynthetic(duration, dt);
    
    std::cout << "Dataset: " << dataset.size() << " samples, " << duration << "s, " 
              << (1.0/dt) << "Hz, noise=" << noise_level*100 << "%" << std::endl;
    printSeparator();
    
    // Test configurations
    std::vector<TestResult> results;
    
    // Test: 9-DOF (Accel + Gyro + Mag)
    std::cout << "Test: 9-DOF (Accelerometer + Gyroscope + Magnetometer)" << std::endl;
    results.push_back(runTest("Madgwick 9-DOF", dataset, ground_truth, true, 0.005));
    results.push_back(runTest("Madgwick 9-DOF (tuned)", dataset, ground_truth, true, 0.001));
    
    for (const auto& r : results) printResult(r);
    printSeparator();
    
    // Summary
    std::cout << "SUMMARY" << std::endl;
    printSeparator();
    std::cout << std::left << std::setw(30) << "Algorithm" 
              << std::right << std::setw(10) << "Roll RMSE"
              << std::setw(10) << "Pitch RMSE"
              << std::setw(10) << "Yaw RMSE"
              << std::setw(12) << "Orient RMSE"
              << std::setw(10) << "Max Err"
              << std::setw(10) << "Conv. (s)"
              << std::setw(12) << "Avg Time(us)" << std::endl;
    printSeparator();
    
    for (const auto& r : results) {
        std::cout << std::left << std::setw(30) << r.algorithm_name
                  << std::right << std::setw(10) << std::fixed << std::setprecision(2) << r.rmse_roll_deg
                  << std::setw(10) << r.rmse_pitch_deg
                  << std::setw(10) << r.rmse_yaw_deg
                  << std::setw(12) << r.rmse_orientation_deg
                  << std::setw(10) << r.max_error_deg
                  << std::setw(10) << r.convergence_time_s
                  << std::setw(12) << r.avg_update_time_us << std::endl;
    }
    
    printSeparator();
    
    // Find best overall
    auto best = std::min_element(results.begin(), results.end(), 
        [](const TestResult& a, const TestResult& b) {
            return a.rmse_orientation_deg < b.rmse_orientation_deg;
        });
    
    std::cout << "Best overall (by orientation RMSE): " << best->algorithm_name 
              << " (" << best->rmse_orientation_deg << " deg)" << std::endl;
    
    return 0;
}
