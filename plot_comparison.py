import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('comparison_9dof.csv')

fig, axes = plt.subplots(4, 2, figsize=(14, 16))

# Euler angles comparison
axes[0, 0].plot(df['timestamp'], df['gt_roll'], 'k-', label='GT', linewidth=1.5)
axes[0, 0].plot(df['timestamp'], df['madgwick_roll'], 'b:', label='Madgwick', alpha=0.7)
axes[0, 0].set_ylabel('Roll (deg)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_title('Roll Comparison')

axes[0, 1].plot(df['timestamp'], df['gt_pitch'], 'k-', label='GT', linewidth=1.5)
axes[0, 1].plot(df['timestamp'], df['madgwick_pitch'], 'b:', label='Madgwick', alpha=0.7)
axes[0, 1].set_ylabel('Pitch (deg)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_title('Pitch Comparison')

axes[1, 0].plot(df['timestamp'], df['gt_yaw'], 'k-', label='GT', linewidth=1.5)
axes[1, 0].plot(df['timestamp'], df['madgwick_yaw'], 'b:', label='Madgwick', alpha=0.7)
axes[1, 0].set_ylabel('Yaw (deg)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_title('Yaw Comparison')

# Error comparison
axes[1, 1].plot(df['timestamp'], df['madgwick_error'], 'b-', label='Madgwick', alpha=0.7)
axes[1, 1].set_ylabel('Orientation Error (deg)')
axes[1, 1].set_xlabel('Time (s)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_title('Orientation Error vs Time')

# Error histogram
axes[2, 0].hist(df['madgwick_error'], bins=50, alpha=0.5, label='Madgwick', color='blue')
axes[2, 0].set_xlabel('Orientation Error (deg)')
axes[2, 0].set_ylabel('Count')
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)
axes[2, 0].set_title('Error Distribution')

# Cumulative RMSE
def cum_rmse(errors):
    return np.sqrt(np.cumsum(errors**2) / (np.arange(len(errors)) + 1))

axes[3, 0].plot(df['timestamp'], cum_rmse(df['madgwick_error']), 'b-', label='Madgwick')
axes[3, 0].set_ylabel('Cumulative RMSE (deg)')
axes[3, 0].set_xlabel('Time (s)')
axes[3, 0].legend()
axes[3, 0].grid(True, alpha=0.3)
axes[3, 0].set_title('Cumulative RMSE')

# Phase plane (Roll error vs Pitch error)
axes[3, 1].scatter(df['madgwick_roll'] - df['gt_roll'], df['madgwick_pitch'] - df['gt_pitch'], 
                   c=df['timestamp'], cmap='Blues', alpha=0.5, s=1, label='Madgwick')
axes[3, 1].set_xlabel('Roll Error (deg)')
axes[3, 1].set_ylabel('Pitch Error (deg)')
axes[3, 1].legend()
axes[3, 1].grid(True, alpha=0.3)
axes[3, 1].set_title('Roll-Pitch Error Phase Plane')

plt.tight_layout()
plt.savefig('comparison_9dof.png', dpi=150)
plt.show()

print("Plotting complete!")
