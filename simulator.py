"""
simulator.py
Real-Time Industrial IoT Sensor Stream Generator & Fault Injection Simulator
Simulates a CNC Milling Machine operating under normal and abnormal conditions.
"""

import time
import random
import numpy as np
import pandas as pd

class IndustrialSensorSimulator:
    def __init__(self):
        # Baseline normal working parameters
        self.air_temp_base = 299.5      # Kelvin (~26.3 C)
        self.process_temp_base = 309.5  # Kelvin (~36.3 C)
        self.rpm_base = 1520.0          # RPM
        self.torque_base = 40.0         # Nm
        self.tool_wear_base = 35.0      # minutes
        self.vibration_base = 1.2       # mm/s (simulated accelerometer)
        
        # State tracking
        self.current_tool_wear = self.tool_wear_base
        self.fault_mode = "NORMAL"
        self.history = []
        self.max_history = 60 # retain last 60 time steps

    def set_fault_mode(self, mode: str):
        """Changes the simulated machine state to trigger specific failures."""
        self.fault_mode = mode.upper()
        if self.fault_mode == "RESET":
            self.fault_mode = "NORMAL"
            self.current_tool_wear = 25.0

    def generate_reading(self, manual_overrides: dict = None):
        """
        Generates a single time-step sensor packet.
        Can receive manual slider overrides for live judge demonstration.
        """
        # Realistic Gaussian process noise
        noise_temp = np.random.normal(0, 0.25)
        noise_rpm = np.random.normal(0, 15)
        noise_torque = np.random.normal(0, 1.2)
        noise_vib = np.random.normal(0, 0.08)
        
        # Gradually increment tool wear in normal operation
        self.current_tool_wear += 0.15

        air_temp = self.air_temp_base + noise_temp
        process_temp = self.process_temp_base + noise_temp
        rpm = self.rpm_base + noise_rpm
        torque = self.torque_base + noise_torque
        tool_wear = self.current_tool_wear
        vibration = self.vibration_base + noise_vib

        # Apply Injected Fault Conditions according to manufacturing physics
        if self.fault_mode == "HEAT_DISSIPATION":
            # HDF occurs when Temp Diff < 8.6 K and Rotational Speed < 1380 RPM
            air_temp += np.random.uniform(3.5, 5.0)
            process_temp = air_temp + np.random.uniform(5.5, 7.5) # narrow temp diff
            rpm = np.random.uniform(1250, 1360)
            vibration += np.random.uniform(1.2, 2.0)

        elif self.fault_mode == "POWER_OVERLOAD":
            # PWF occurs when Power (Torque * RPM * 2*pi/60) exceeds 9000 W or drops < 3500 W
            torque = np.random.uniform(68.0, 78.0) # extreme torque
            rpm = np.random.uniform(1380, 1490)    # high mechanical load
            vibration += np.random.uniform(2.5, 4.0)

        elif self.fault_mode == "OVERSTRAIN":
            # OSF occurs when Tool Wear * Torque exceeds critical threshold (> 11000)
            tool_wear = np.random.uniform(205.0, 240.0)
            torque = np.random.uniform(56.0, 68.0)
            rpm = np.random.uniform(1320, 1450)
            vibration += np.random.uniform(2.2, 3.8)

        elif self.fault_mode == "TOOL_WEAR":
            # TWF occurs when tool wear exceeds 210-240 minutes
            tool_wear = np.random.uniform(225.0, 255.0)
            torque += np.random.uniform(6.0, 12.0)
            vibration += np.random.uniform(1.8, 3.0)

        elif self.fault_mode == "ANOMALY_DRIFT":
            # Unsupervised anomaly: unusual combination of high RPM and fluctuating torque
            rpm = np.random.uniform(2300, 2600)
            torque = np.random.uniform(14.0, 20.0)
            vibration = np.random.uniform(3.2, 5.2)

        # Apply manual judge overrides if provided
        if manual_overrides:
            if 'Air_temp_K' in manual_overrides and manual_overrides['Air_temp_K'] is not None:
                air_temp = float(manual_overrides['Air_temp_K'])
            if 'Process_temp_K' in manual_overrides and manual_overrides['Process_temp_K'] is not None:
                process_temp = float(manual_overrides['Process_temp_K'])
            if 'Rot_speed_rpm' in manual_overrides and manual_overrides['Rot_speed_rpm'] is not None:
                rpm = float(manual_overrides['Rot_speed_rpm'])
            if 'Torque_Nm' in manual_overrides and manual_overrides['Torque_Nm'] is not None:
                torque = float(manual_overrides['Torque_Nm'])
            if 'Tool_wear_min' in manual_overrides and manual_overrides['Tool_wear_min'] is not None:
                tool_wear = float(manual_overrides['Tool_wear_min'])

        # Calculate physical derived features
        temp_diff = process_temp - air_temp
        power_watts = torque * (rpm * 2.0 * np.pi / 60.0)
        overstrain_factor = torque * tool_wear
        temp_speed_ratio = process_temp / (rpm + 1e-5)

        reading = {
            'timestamp': time.strftime("%H:%M:%S"),
            'Air_temp_K': round(air_temp, 2),
            'Process_temp_K': round(process_temp, 2),
            'Rot_speed_rpm': round(rpm, 1),
            'Torque_Nm': round(torque, 2),
            'Tool_wear_min': round(tool_wear, 1),
            'Vibration_mms': round(max(0.2, vibration), 2),
            'Temp_diff_K': round(temp_diff, 2),
            'Power_Watts': round(power_watts, 1),
            'Overstrain_factor': round(overstrain_factor, 1),
            'Temp_Speed_ratio': round(temp_speed_ratio, 5),
            'injected_mode': self.fault_mode
        }

        self.history.append(reading)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        return reading

    def get_history_dataframe(self):
        """Returns the rolling historical telemetry as a Pandas DataFrame."""
        if not self.history:
            return pd.DataFrame()
        return pd.DataFrame(self.history)
