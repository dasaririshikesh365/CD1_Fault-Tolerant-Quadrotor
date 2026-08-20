# Fault-Tolerant Data Fusion for Quadrotor UAV using MuJoCo

## Project Overview

This project presents the design and implementation of a fault-tolerant data fusion system for a quadrotor UAV, developed and validated entirely in a simulation environment using MuJoCo.

The system focuses on maintaining accurate estimation of the UAV’s position and orientation even in the presence of faults affecting sensors or estimation algorithms. The work is inspired by the methodology described in:

*“Data fusion fault tolerant strategy for a quadrotor UAV under sensors and software faults”*

The original paper proposes a fault-tolerant architecture combining data fusion, redundancy, and voting mechanisms. In this project, those concepts are adapted and implemented in a purely simulated setup.

---

## Problem Statement

Quadrotor UAVs rely on multiple sensors such as IMU, GPS, and magnetometers to estimate their state. However, these systems are prone to:

* Sensor faults (noise spikes, freezing, incorrect readings)
* Software faults (incorrect initialization, filter divergence)

Such faults can degrade estimation accuracy and lead to instability.

This project addresses the problem by designing a system that:

* Detects inconsistencies in sensor data
* Identifies faulty components
* Maintains reliable estimation using redundancy and fusion

---

## Implementation Environment

### Simulation Platform

The entire system is implemented using MuJoCo (Multi-Joint dynamics with Contact), which provides:

* Physics-based simulation of quadrotor dynamics
* Realistic motion modeling using rigid-body equations
* Controlled environment for testing fault scenarios

### Nature of Implementation

* Fully simulation-based
* No hardware integration
* Sensor data is generated or derived from simulation states
* Faults are injected programmatically

---

## System Architecture

The architecture is inspired by the fault-tolerant structure described in the paper, which includes multiple estimation branches and a decision mechanism.

### Sensor Simulation

Instead of real sensors, the system uses simulated inputs corresponding to:

* IMU (acceleration and angular velocity)
* GPS (position)
* Magnetometer (orientation reference)

Noise and faults are artificially introduced to mimic real-world behavior.

---

### Data Fusion Modules

Multiple estimation branches are implemented using Kalman Filters / Extended Kalman Filters.

Each branch:

* Receives sensor inputs
* Produces an independent estimate of the UAV state

This redundancy is critical for fault detection.

---

### Dynamic Model (Analytical Redundancy)

A mathematical model of the quadrotor is used as an additional estimation source.

According to the paper, this model acts as a reference and helps generate residuals by comparing predicted and measured outputs.

This provides an additional layer of validation beyond sensor data.

---

### Fault Detection Mechanism

Fault detection is based on residual analysis:

Residual = Measured Value – Estimated Value

* Small residual → normal behavior
* Large residual → potential fault

This approach aligns with the use of residuals in Kalman filter-based fault detection described in the paper.

---

### Voting Mechanism

A weighted voting system is used to combine outputs from:

* Multiple Kalman filter branches
* Dynamic model estimation

Instead of simple averaging, weights are assigned based on consistency between estimates.

As described in the paper, this improves:

* Accuracy of final output
* Fault detection capability

---

### Fault Identification and Recovery

Once a fault is detected:

* The faulty branch is identified based on inconsistency
* Its influence is reduced or removed
* The system continues using reliable estimates

This ensures continuous operation without failure propagation.

---

## UAV Model

The quadrotor is modeled using simplified dynamics based on Newton–Euler equations.

Key assumptions (as stated in the paper):

* Symmetrical rigid body
* Small-angle motion
* Thrust proportional to square of rotor speed
* Negligible motor dynamics

These assumptions simplify simulation while preserving essential behavior.

---

## Simulation Workflow

1. Initialize UAV state in MuJoCo
2. Generate simulated sensor readings
3. Apply Kalman Filter prediction step
4. Apply correction using sensor data
5. Inject faults into selected sensors or modules
6. Compute residuals
7. Detect inconsistencies
8. Apply voting mechanism
9. Output corrected state estimate

---

## Key Features

* Simulation of both sensor and software faults
* Multi-branch data fusion using EKF
* Analytical redundancy using dynamic model
* Weighted voting-based decision system
* Fault detection and recovery without system shutdown

---

## Results and Observations

From simulation experiments:

* The system maintains stable estimation even under faulty conditions
* Faulty sensors are successfully detected through residual analysis
* Weighted voting improves robustness compared to simple averaging
* Analytical redundancy provides an effective fallback mechanism

---

## Limitations

* Only single or sequential faults are considered
* Highly complex real-world disturbances are not modeled
* Accuracy depends on model assumptions and tuning parameters

---

## Future Work

* Integration with real UAV hardware
* Use of advanced filters (UKF, Particle Filter)
* Machine learning-based fault detection
* Multi-UAV fault-tolerant coordination
* More realistic environmental disturbances

---

## Conclusion

This project demonstrates that a fault-tolerant UAV estimation system can be effectively designed and validated using simulation alone. By combining data fusion, redundancy, and voting mechanisms, the system achieves reliable performance even in the presence of faults.

The use of MuJoCo allows safe testing of failure scenarios that would be difficult or risky in real-world setups.

---

## Team Members

Palaparthi Sathvik — CB.SC.U4AIE24240
Rishi — CB.SC.U4AIE24365
Saradhi — CB.SC.U4AIE24265
Aryan — CB.SC.U4AIE24341

---

## Reference

H. Hamadi, B. Lussier, I. Fantoni, C. Francis,
“Data fusion fault tolerant strategy for a quadrotor UAV under sensors and software faults,”
ISA Transactions, 2022.

---

