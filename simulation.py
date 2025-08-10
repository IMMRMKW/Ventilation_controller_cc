import matplotlib.pyplot as plt
import numpy as np
import math 

def determine_index(sensor_value, index_list):
    """
    Returns:
        idx (float): fractional index, i + ratio, where ratio is how far sensor_value is between index_list[i] and index_list[i+1]
    """

    for i in range(len(index_list) - 1):
        if sensor_value <= index_list[i]:
            ratio = sensor_value / index_list[i]
            return ratio
        if index_list[i] < sensor_value <= index_list[i + 1]:
            ratio = (sensor_value - index_list[i]) / (index_list[i + 1] - index_list[i])
            return i + ratio + 1
    # If above last threshold, return last index
    return float(len(index_list) - 1)

# time step (s)
dt = 1.0

# time constants (s)
tau_up   = 15    # fast attack
tau_down = 60   # slow release

# compute smoothing factors
alpha_up   = 1 - math.exp(-dt / tau_up)
alpha_down = 1 - math.exp(-dt / tau_down)
print(f"alpha_up: {alpha_up}, alpha_down: {alpha_down}")

# initial filtered index
y = 0.0

def asymmetric_lpf(x):
    global y
    if x > y:
        a = alpha_up
    else:
        a = alpha_down

    # update
    y += a * (x - y)
    return y


# Simulation parameters
dt          = 1.0                # time step [s]
total_time  = 3600               # total duration [s]
x_rate      = 8               # CO₂ generation rate [ppm/s]
setpoint   = 0.5              # air quality setpoint
initial_co2 = 600.0              # starting CO₂ [ppm]



# Fan discrete speeds [ppm/s]
fan_speeds = np.round(np.arange(4.4, 8.9 + 1e-6, 0.1), 1)

# PID coefficients (tune as needed)
fan_diff = fan_speeds.max()-fan_speeds.min()
Kp = (fan_speeds.max()-fan_speeds.min()) / 5
Ki = [fan_diff / 30, fan_diff / 15, fan_diff / 7.5, fan_diff / 3.75, fan_diff / 1.875, fan_diff / 0.9375]
Kd = 0.005

# Storage arrays
times     = np.arange(0, total_time + dt, dt)
co2_level = np.zeros_like(times)
fan_rate  = np.zeros_like(times)
air_quality_index = np.zeros_like(times)
filtered_index = np.zeros_like(times)

# Initialize
co2_level[0] = initial_co2
air_quality_index[0] = 0
filtered_index[0] = 0.0

integral     = 0.0
prev_error   = 0.0

co2_index = [650, 1500, 2000, 2500, 5000]

# Simulation loop
for i in range(0, len(times)):
    # Natural CO₂ build-up
    co2 = co2_level[i-1] + x_rate * dt

    air_quality_index[i] = determine_index(co2, co2_index)
    filtered_index[i] = asymmetric_lpf(air_quality_index[i])
    Ki_index_based = Ki[int(round(filtered_index[i]))]
    # Compute PID error (positive when CO₂ > threshold)
    error       = air_quality_index[i] - setpoint
    integral   += error*Ki_index_based
    derivative  = (error - prev_error) / dt
    prev_error  = error

    # PID output (desired removal rate)
    pid_output = Kp * error + integral + fan_speeds.min() # + Kd * derivative

    # Map to discrete fan speed:
    #  - if pid_output < minimum speed, fan stays off (0)
    #  - else clamp to [4.4, 8.9] and round to nearest 0.1
    if pid_output < fan_speeds[0]:
        rate = fan_speeds[0]
        integral   -= error*Ki_index_based
    if pid_output > fan_speeds[-1]:
        rate = fan_speeds[-1]
        integral   -= error*Ki_index_based
    else:
        rate = float(np.clip(pid_output, fan_speeds[0], fan_speeds[-1]))
        rate = float(np.round(rate, 1))
    

    # Apply removal
    co2 = max(co2 - rate * dt, 0.0)

    # Record
    co2_level[i] = co2
    fan_rate[i]  = rate

# Plotting
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

ax1.plot(times, air_quality_index, label="air_quality_index", color='blue')
ax1.plot(times, filtered_index, label="fitlered_index", color='red')
ax1.axhline(setpoint, color='r', linestyle='--', label="Threshold (800 ppm)")
ax1.set_ylabel("Air Quality Index")
ax1.legend()
ax1.grid(True)

ax2.step(times, fan_rate, where='post', label="Fan Removal Rate (ppm/s)")
ax2.set_ylabel("Fan Rate (ppm/s)")
ax2.set_xlabel("Time (s)")
ax2.legend()
ax2.grid(True)

ax3.plot(times, co2_level, label="air_quality_index", color='blue')
ax3.plot(times, filtered_index, label="fitlered_index", color='red')
ax3.set_ylabel("CO₂ Level / ppm")
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.show()

