#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include "ImuData.h"
#include "MadgwickAHRS.h"
#include "QuaternionUtils.h"

using namespace QuaternionUtils;

struct ComparisonData {
    double timestamp;
    double madgwick_roll, madgwick_pitch, madgwick_yaw;
    double gt_roll, gt_pitch, gt_yaw;
    double madgwick_error;
};

int main(int argc, char* argv[]) {
    // Parse arguments
    bool use_magnetometer = true;
    double duration = 20.0;
    double dt = 0.01;
    double noise_level = 0.02;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--no-mag") use_magnetometer = false;
        else if (arg == "--duration") duration = std::stod(argv[++i]);
        else if (arg == "--dt") dt = std::stod(argv[++i]);
        else if (arg == "--noise") noise_level = std::stod(argv[++i]);
    }
    
    std::cout << "IMU AHRS Detailed Comparison: GT vs Madgwick" << std::endl;
    std::cout << "=================================================" << std::endl;
    std::cout << "Duration: " << duration << "s, dt: " << dt << "s (" << 1.0/dt << "Hz)" << std::endl;
    std::cout << "Noise level: " << noise_level * 100 << "%" << std::endl;
    std::cout << "Magnetometer: " << (use_magnetometer ? "Enabled" : "Disabled") << std::endl;
    std::cout << std::endl;
    
    // Generate test data
    ImuDataset dataset = ImuDataset::generateSynthetic(duration, dt, noise_level);
    GroundTruthDataset ground_truth = GroundTruthDataset::generateSynthetic(duration, dt);
    
    // Configure filters    
    MadgwickAHRS::Config madgwick_config;
    madgwick_config.beta = use_magnetometer ? 0.08 : 0.05;
    madgwick_config.use_magnetometer = use_magnetometer;
    madgwick_config.adaptive_beta = true;
    madgwick_config.accel_threshold = 0.2;
    MadgwickAHRS madgwick(madgwick_config);
    madgwick.init(ground_truth[0].q);
    
    // Run comparison
    std::vector<ComparisonData> data;
    data.reserve(dataset.size());
    
    double madgwick_total_time = 0;
    double sum_sq_madgwick = 0;
    double max_madgwick = 0;
    int conv_madgwick = -1;
    int conv_count_g = 0;
    
    for (size_t i = 0; i < dataset.size(); ++i) {
        const auto& imu = dataset[i];
        const auto& gt = ground_truth[i];
        
        // Madgwick update
        auto start = std::chrono::high_resolution_clock::now();
        madgwick.update(imu, dt);
        auto end = std::chrono::high_resolution_clock::now();
        madgwick_total_time += std::chrono::duration<double, std::micro>(end - start).count();
        
        // Get estimates
        Eigen::Vector3d euler_g = toEulerZYX(madgwick.getOrientation());
        Eigen::Vector3d euler_t = toEulerZYX(gt.q);
        
        // Compute errors
        double err_g = orientationErrorDeg(madgwick.getOrientation(), gt.q);
        
        sum_sq_madgwick += err_g * err_g;
        max_madgwick = std::max(max_madgwick, err_g);
        
        // Convergence detection (error < 2 deg for 1 second)
        if (err_g < 2.0) conv_count_g++; else conv_count_g = 0;
        if (conv_count_g >= 100 && conv_madgwick == -1) conv_madgwick = i;
        
        // Store data for CSV export
        ComparisonData d;
        d.timestamp = imu.timestamp;
        d.madgwick_roll = euler_g.x() * 180.0 / M_PI;
        d.madgwick_pitch = euler_g.y() * 180.0 / M_PI;
        d.madgwick_yaw = euler_g.z() * 180.0 / M_PI;
        d.gt_roll = euler_t.x() * 180.0 / M_PI;
        d.gt_pitch = euler_t.y() * 180.0 / M_PI;
        d.gt_yaw = euler_t.z() * 180.0 / M_PI;
        d.madgwick_error = err_g;
        data.push_back(d);
    }
    
    // Compute statistics
    int n = dataset.size();
    double rmse_g = std::sqrt(sum_sq_madgwick / n);
    
    // Compute per-axis RMSE
    double sum_sq_g_r = 0, sum_sq_g_p = 0, sum_sq_g_y = 0;
    
    for (const auto& d : data) {
        double dr_g = d.madgwick_roll - d.gt_roll;
        double dp_g = d.madgwick_pitch - d.gt_pitch;
        double dy_g = d.madgwick_yaw - d.gt_yaw;
        if (dy_g > 180) dy_g -= 360;
        if (dy_g < -180) dy_g += 360;

        sum_sq_g_r += dr_g * dr_g;
        sum_sq_g_p += dp_g * dp_g;
        sum_sq_g_y += dy_g * dy_g;
    }
    
    // Print results
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "RESULTS" << std::endl;
    std::cout << "=======" << std::endl;
    std::cout << std::endl;
    std::cout << "Overall Orientation RMSE:" << std::endl;
    std::cout << "  Madgwick: " << rmse_g << " deg" << std::endl;
    std::cout << std::endl;
    std::cout << "Per-Axis RMSE:" << std::endl;
    std::cout << "           Roll      Pitch     Yaw" << std::endl;
    std::cout << "  Madgwick:" << std::setw(8) << std::sqrt(sum_sq_g_r/n) 
              << "  " << std::setw(8) << std::sqrt(sum_sq_g_p/n)
              << "  " << std::setw(8) << std::sqrt(sum_sq_g_y/n) << std::endl;
    std::cout << std::endl;
    std::cout << "Max Error:" << std::endl;
    std::cout << "  Madgwick: " << max_madgwick << " deg" << std::endl;
    std::cout << std::endl;
    std::cout << "Convergence Time (error < 2 deg for 1s):" << std::endl;
    std::cout << "  Madgwick: " << (conv_madgwick >= 0 ? std::to_string(conv_madgwick * dt) + " s" : "Not converged") << std::endl;
    std::cout << std::endl;
    std::cout << "Average Update Time:" << std::endl;
    std::cout << "  Madgwick: " << madgwick_total_time / n << " us" << std::endl;
    std::cout << std::endl;
    
    // Export CSV for plotting
    std::string csv_file = use_magnetometer ? "comparison_9dof.csv" : "comparison_6dof.csv";
    std::ofstream csv(csv_file);
    csv << "timestamp,madgwick_roll,madgwick_pitch,madgwick_yaw,gt_roll,gt_pitch,gt_yaw,madgwick_error\n";
    for (const auto& d : data) {
        csv << d.timestamp << ","
            << d.madgwick_roll << "," << d.madgwick_pitch << "," << d.madgwick_yaw << ","
            << d.gt_roll << "," << d.gt_pitch << "," << d.gt_yaw << ","
            << d.madgwick_error << "\n";
    }
    csv.close();
    std::cout << "Exported data to " << csv_file << std::endl;
    
    // Export Python plotting script
    std::string py_file = "plot_comparison.py";
    std::ofstream py(py_file);
    py << "import pandas as pd\n";
    py << "import matplotlib.pyplot as plt\n";
    py << "import numpy as np\n\n";
    py << "df = pd.read_csv('" << csv_file << "')\n\n";
    py << "fig, axes = plt.subplots(4, 2, figsize=(14, 16))\n\n";
    py << "# Euler angles comparison\n";
    py << "axes[0, 0].plot(df['timestamp'], df['gt_roll'], 'k-', label='GT', linewidth=1.5)\n";
    py << "axes[0, 0].plot(df['timestamp'], df['madgwick_roll'], 'b:', label='Madgwick', alpha=0.7)\n";
    py << "axes[0, 0].set_ylabel('Roll (deg)')\n";
    py << "axes[0, 0].legend()\n";
    py << "axes[0, 0].grid(True, alpha=0.3)\n";
    py << "axes[0, 0].set_title('Roll Comparison')\n\n";
    py << "axes[0, 1].plot(df['timestamp'], df['gt_pitch'], 'k-', label='GT', linewidth=1.5)\n";
    py << "axes[0, 1].plot(df['timestamp'], df['madgwick_pitch'], 'b:', label='Madgwick', alpha=0.7)\n";
    py << "axes[0, 1].set_ylabel('Pitch (deg)')\n";
    py << "axes[0, 1].legend()\n";
    py << "axes[0, 1].grid(True, alpha=0.3)\n";
    py << "axes[0, 1].set_title('Pitch Comparison')\n\n";
    py << "axes[1, 0].plot(df['timestamp'], df['gt_yaw'], 'k-', label='GT', linewidth=1.5)\n";
    py << "axes[1, 0].plot(df['timestamp'], df['madgwick_yaw'], 'b:', label='Madgwick', alpha=0.7)\n";
    py << "axes[1, 0].set_ylabel('Yaw (deg)')\n";
    py << "axes[1, 0].legend()\n";
    py << "axes[1, 0].grid(True, alpha=0.3)\n";
    py << "axes[1, 0].set_title('Yaw Comparison')\n\n";
    py << "# Error comparison\n";
    py << "axes[1, 1].plot(df['timestamp'], df['madgwick_error'], 'b-', label='Madgwick', alpha=0.7)\n";
    py << "axes[1, 1].set_ylabel('Orientation Error (deg)')\n";
    py << "axes[1, 1].set_xlabel('Time (s)')\n";
    py << "axes[1, 1].legend()\n";
    py << "axes[1, 1].grid(True, alpha=0.3)\n";
    py << "axes[1, 1].set_title('Orientation Error vs Time')\n\n";
    py << "# Error histogram\n";
    py << "axes[2, 0].hist(df['madgwick_error'], bins=50, alpha=0.5, label='Madgwick', color='blue')\n";
    py << "axes[2, 0].set_xlabel('Orientation Error (deg)')\n";
    py << "axes[2, 0].set_ylabel('Count')\n";
    py << "axes[2, 0].legend()\n";
    py << "axes[2, 0].grid(True, alpha=0.3)\n";
    py << "axes[2, 0].set_title('Error Distribution')\n\n";
    py << "# Cumulative RMSE\n";
    py << "def cum_rmse(errors):\n";
    py << "    return np.sqrt(np.cumsum(errors**2) / (np.arange(len(errors)) + 1))\n\n";
    py << "axes[3, 0].plot(df['timestamp'], cum_rmse(df['madgwick_error']), 'b-', label='Madgwick')\n";
    py << "axes[3, 0].set_ylabel('Cumulative RMSE (deg)')\n";
    py << "axes[3, 0].set_xlabel('Time (s)')\n";
    py << "axes[3, 0].legend()\n";
    py << "axes[3, 0].grid(True, alpha=0.3)\n";
    py << "axes[3, 0].set_title('Cumulative RMSE')\n\n";
    py << "# Phase plane (Roll error vs Pitch error)\n";
    py << "axes[3, 1].scatter(df['madgwick_roll'] - df['gt_roll'], df['madgwick_pitch'] - df['gt_pitch'], \n";
    py << "                   c=df['timestamp'], cmap='Blues', alpha=0.5, s=1, label='Madgwick')\n";
    py << "axes[3, 1].set_xlabel('Roll Error (deg)')\n";
    py << "axes[3, 1].set_ylabel('Pitch Error (deg)')\n";
    py << "axes[3, 1].legend()\n";
    py << "axes[3, 1].grid(True, alpha=0.3)\n";
    py << "axes[3, 1].set_title('Roll-Pitch Error Phase Plane')\n\n";
    py << "plt.tight_layout()\n";
    py << "plt.savefig('" << (use_magnetometer ? "comparison_9dof.png" : "comparison_6dof.png") << "', dpi=150)\n";
    py << "plt.show()\n\n";
    py << "print(\"Plotting complete!\")\n";
    py.close();
    std::cout << "Exported Python plotting script to " << py_file << std::endl;
    
    return 0;
}
