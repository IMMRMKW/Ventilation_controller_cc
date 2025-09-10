"""Constants for the PID Ventilation Control integration."""

DOMAIN = "ventilation"

# Dispatcher signal base for AQI updates (per entry: f"{SIGNAL_AQI_UPDATED}_{entry_id}")
SIGNAL_AQI_UPDATED = f"{DOMAIN}_aqi_updated"
SIGNAL_RATE_UPDATED = f"{DOMAIN}_rate_updated"

# Default CO2 thresholds for air quality index
DEFAULT_CO2_INDEX = [400, 650, 1500, 2000, 2500, 5000]
DEFAULT_VOC_PPM_INDEX = [0, 15, 25, 50, 75, 100]
DEFAULT_VOC_INDEX = [0, 300, 500, 1000, 3000, 5000]
DEFAULT_PM_1_0_INDEX = [0, 12, 35, 55, 150, 250]
DEFAULT_PM_2_5_INDEX = [0, 12, 35, 55, 150, 250]
DEFAULT_PM_10_INDEX = [0, 54, 154, 254, 354, 424]
DEFAULT_HUMIDITY_INDEX = [5, 10, 15, 20, 25, 30]

# Configuration keys
CONF_CO2_SENSORS = "co2_sensors"
CONF_VOC_SENSORS = "voc_sensors"
CONF_PM_SENSORS = "pm_sensors"
CONF_HUMIDITY_SENSORS = "humidity_sensors"
CONF_SETPOINT = "setpoint"
CONF_MIN_FAN_OUTPUT = "min_fan_output"
CONF_MAX_FAN_OUTPUT = "max_fan_output"
CONF_KP = "kp"
CONF_KI_TIMES = "ki_times"
CONF_UPDATE_INTERVAL = "update_interval"
CONF_REMOTE_DEVICE = "remote_device"
CONF_FAN_DEVICE = "fan_device"
CONF_VALVE_DEVICES = "valve_devices"
CONF_NUM_ZONES = "num_zones"
CONF_CO2_SENSOR_ZONES = "co2_sensor_zones"
CONF_VOC_SENSOR_ZONES = "voc_sensor_zones"
CONF_PM_SENSOR_ZONES = "pm_sensor_zones"
CONF_HUMIDITY_SENSOR_ZONES = "humidity_sensor_zones"
CONF_ZONE_CONFIGS = "zone_configs"
CONF_ZONE_ID_FROM = "id_from"
CONF_ZONE_ID_TO = "id_to"
CONF_ZONE_SENSOR_ID = "sensor_id"
CONF_ZONE_MIN_FAN = "min_fan_rate"
CONF_ZONE_MAX_FAN = "max_fan_rate"
CONF_CO2_INDEX = "co2_index"
CONF_VOC_INDEX = "voc_index"
CONF_VOC_PPM_INDEX = "voc_ppm_index"
CONF_PM_1_0_INDEX = "pm_1_0_index"
CONF_PM_2_5_INDEX = "pm_2_5_index"
CONF_PM_10_INDEX = "pm_10_index"
CONF_HUMIDITY_INDEX = "humidity_index"
