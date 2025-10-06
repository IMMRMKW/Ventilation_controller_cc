from datetime import timedelta, datetime
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.const import Platform
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
import logging
import numpy as np
import time

from .const import (
    DOMAIN,
    SIGNAL_AQI_UPDATED,
    SIGNAL_RATE_UPDATED,
    DEFAULT_CO2_INDEX,
    DEFAULT_VOC_INDEX,
    DEFAULT_VOC_PPM_INDEX,
    DEFAULT_PM_1_0_INDEX,
    DEFAULT_PM_2_5_INDEX,
    DEFAULT_PM_10_INDEX,
    DEFAULT_HUMIDITY_INDEX,
    CONF_CO2_SENSORS,
    CONF_VOC_SENSORS,
    CONF_PM_SENSORS,
    CONF_HUMIDITY_SENSORS,
    CONF_SETPOINT,
    CONF_MIN_FAN_OUTPUT,
    CONF_MAX_FAN_OUTPUT,
    CONF_KP,
    CONF_KI_TIMES,
    CONF_UPDATE_INTERVAL,
    CONF_REMOTE_DEVICE,
    CONF_FAN_DEVICE,
    CONF_VALVE_DEVICES,
    CONF_NUM_ZONES,
    CONF_ZONE_CONFIGS,
    CONF_ZONE_ID_FROM,
    CONF_ZONE_ID_TO,
    CONF_ZONE_SENSOR_ID,
    CONF_ZONE_MIN_FAN,
    CONF_ZONE_MAX_FAN,
    CONF_CO2_SENSOR_ZONES,
    CONF_VOC_SENSOR_ZONES,
    CONF_PM_SENSOR_ZONES,
    CONF_HUMIDITY_SENSOR_ZONES,
    CONF_CO2_INDEX,
    CONF_VOC_INDEX,
    CONF_VOC_PPM_INDEX,
    CONF_PM_1_0_INDEX,
    CONF_PM_2_5_INDEX,
    CONF_PM_10_INDEX,
    CONF_HUMIDITY_INDEX,
)

_LOGGER = logging.getLogger(__name__)

PLATFORMS = [Platform.SWITCH, Platform.SENSOR, Platform.NUMBER]

class HumidityMovingAverage:
    """Manages 24-hour moving averages for humidity sensors."""
    
    def __init__(self):
        """Initialize the humidity moving average tracker."""
        self.sensor_data: dict[str, list[tuple[datetime, float]]] = {}
        self._cleanup_interval = timedelta(hours=1)  # Clean up old data every hour
        self._last_cleanup = datetime.now()
    
    def add_reading(self, sensor_id: str, value: float, timestamp: datetime = None) -> None:
        """Add a new humidity reading for a sensor."""
        if timestamp is None:
            timestamp = datetime.now()
            
        # Initialize sensor data if not exists
        if sensor_id not in self.sensor_data:
            self.sensor_data[sensor_id] = []
            
        # Add new reading
        self.sensor_data[sensor_id].append((timestamp, value))
        
        # Clean up old data periodically
        if datetime.now() - self._last_cleanup > self._cleanup_interval:
            self._cleanup_old_data()
            self._last_cleanup = datetime.now()
    
    def get_24h_average(self, sensor_id: str) -> float | None:
        """Get the 24-hour moving average for a sensor."""
        if sensor_id not in self.sensor_data:
            return None
            
        now = datetime.now()
        cutoff_time = now - timedelta(hours=24)
        
        # Filter readings from last 24 hours
        recent_readings = [
            (timestamp, value) for timestamp, value in self.sensor_data[sensor_id]
            if timestamp >= cutoff_time
        ]
        
        if not recent_readings:
            return None
            
        # Calculate weighted average based on time intervals
        if len(recent_readings) == 1:
            return recent_readings[0][1]
            
        # Calculate time-weighted average
        total_weighted_value = 0.0
        total_time = 0.0
        
        for i in range(len(recent_readings) - 1):
            current_time, current_value = recent_readings[i]
            next_time, _ = recent_readings[i + 1]
            
            # Time interval this reading was valid for
            interval = (next_time - current_time).total_seconds()
            total_weighted_value += current_value * interval
            total_time += interval
        
        # Add the last reading weighted to current time
        last_time, last_value = recent_readings[-1]
        final_interval = (now - last_time).total_seconds()
        total_weighted_value += last_value * final_interval
        total_time += final_interval
        
        return total_weighted_value / total_time if total_time > 0 else None
    
    def _cleanup_old_data(self) -> None:
        """Remove readings older than 25 hours (keep a bit extra for safety)."""
        cutoff_time = datetime.now() - timedelta(hours=25)
        
        for sensor_id in self.sensor_data:
            self.sensor_data[sensor_id] = [
                (timestamp, value) for timestamp, value in self.sensor_data[sensor_id]
                if timestamp >= cutoff_time
            ]

# Helper to read sensor state with fallback to previous known good value
def _read_value(hass: HomeAssistant, entity_id: str, previous_values: dict[str, float], fallback: float) -> float:
    state = hass.states.get(entity_id)
    try:
        value = float(state.state)  # type: ignore[attr-defined]
        previous_values[entity_id] = value
        return value
    except Exception:
        return previous_values.get(entity_id, fallback)
    
def determine_index(sensor_value, index_list):
    if sensor_value <= index_list[0]:
        return 0.0
    for i in range(len(index_list) - 1):
        if index_list[i] < sensor_value <= index_list[i + 1]:
            ratio = (sensor_value - index_list[i]) / (index_list[i + 1] - index_list[i])
            return i + ratio
    return float(len(index_list) - 1)

def _worst_air_quality_index(
    hass: HomeAssistant,
    co2_sensors: list[str],
    voc_sensors: list[str],
    pm_sensors: list[str],
    humidity_sensors: list[str],
    thresholds_by_dc: dict[str, list[float]],
    previous_values: dict[str, float],
    humidity_avg_tracker: HumidityMovingAverage,
) -> tuple[float, str, float, str]:
    """Return (max_index, worst_entity_id, value, device_class)."""
    worst_idx = 0.0
    worst_entity = ""
    worst_value = 0.0
    worst_dc = ""

    # Process CO2, VOC, and PM sensors (unchanged logic)
    for eid in list(co2_sensors) + list(voc_sensors) + list(pm_sensors):
        state = hass.states.get(eid)
        dc = state.attributes.get("device_class") if state else None  # type: ignore[union-attr]

        thresholds = thresholds_by_dc.get(dc) if isinstance(dc, str) else None
        if thresholds is None:
            # Fallback by group when device_class missing
            if eid in co2_sensors:
                thresholds = thresholds_by_dc.get("carbon_dioxide", DEFAULT_CO2_INDEX)
            elif eid in voc_sensors:
                thresholds = thresholds_by_dc.get("volatile_organic_compounds", DEFAULT_VOC_INDEX)
            elif eid in pm_sensors:
                thresholds = thresholds_by_dc.get("pm25", DEFAULT_PM_2_5_INDEX)

        # Choose a reasonable fallback value per class
        fallback = 0.0
        if thresholds is DEFAULT_CO2_INDEX or (isinstance(dc, str) and dc == "carbon_dioxide"):
            fallback = 400.0

        value = _read_value(hass, eid, previous_values, fallback)
        try:
            idx = determine_index(value, thresholds)  # type: ignore[arg-type]
        except Exception:
            continue

        if idx > worst_idx:
            worst_idx = idx
            worst_entity = eid
            worst_value = value
            worst_dc = dc or "unknown"

    # Process humidity sensors with 24-hour average deviation logic
    for eid in humidity_sensors:
        state = hass.states.get(eid)
        dc = state.attributes.get("device_class") if state else "humidity"  # type: ignore[union-attr]
        
        thresholds = thresholds_by_dc.get("humidity", DEFAULT_HUMIDITY_INDEX)
        fallback = 50.0  # Reasonable humidity fallback
        
        current_value = _read_value(hass, eid, previous_values, fallback)
        
        # Add current reading to moving average tracker
        humidity_avg_tracker.add_reading(eid, current_value)
        
        # Get 24-hour average
        avg_24h = humidity_avg_tracker.get_24h_average(eid)
        
        if avg_24h is not None:
            # Calculate deviation from 24-hour average
            deviation = abs(current_value - avg_24h)
            
            # Use the deviation against the original thresholds
            # This way, we measure how much the current humidity deviates from the baseline
            try:
                idx = determine_index(deviation, thresholds)
            except Exception:
                continue
            
            _LOGGER.debug(f"Humidity sensor {eid}: current={current_value:.1f}%, 24h_avg={avg_24h:.1f}%, deviation={deviation:.1f}%, index={idx:.2f}")
        else:
            # No 24h average yet, use current value against original thresholds
            try:
                idx = determine_index(current_value, thresholds)
            except Exception:
                continue
            
            _LOGGER.debug(f"Humidity sensor {eid}: current={current_value:.1f}%, no 24h average yet, index={idx:.2f}")

        if idx > worst_idx:
            worst_idx = idx
            worst_entity = eid
            worst_value = current_value
            worst_dc = dc or "humidity"

    return worst_idx, worst_entity, worst_value, worst_dc


def _calculate_zone_air_quality(
    hass: HomeAssistant,
    zone_id: int,
    co2_sensors: list[str],
    voc_sensors: list[str], 
    pm_sensors: list[str],
    humidity_sensors: list[str],
    co2_zones: list[int],
    voc_zones: list[int],
    pm_zones: list[int], 
    humidity_zones: list[int],
    thresholds_by_dc: dict[str, list[float]],
    previous_values: dict[str, float],
    humidity_avg_tracker: HumidityMovingAverage,
) -> tuple[float, str, float, str]:
    """Calculate air quality index for a specific zone using only sensors assigned to that zone."""
    
    # Filter sensors assigned to this zone
    zone_co2_sensors = []
    zone_voc_sensors = []
    zone_pm_sensors = []
    zone_humidity_sensors = []
    
    # Filter CO2 sensors for this zone
    for i, sensor in enumerate(co2_sensors):
        if i < len(co2_zones) and co2_zones[i] == zone_id:
            zone_co2_sensors.append(sensor)
    
    # Filter VOC sensors for this zone        
    for i, sensor in enumerate(voc_sensors):
        if i < len(voc_zones) and voc_zones[i] == zone_id:
            zone_voc_sensors.append(sensor)
            
    # Filter PM sensors for this zone
    for i, sensor in enumerate(pm_sensors):
        if i < len(pm_zones) and pm_zones[i] == zone_id:
            zone_pm_sensors.append(sensor)
            
    # Filter humidity sensors for this zone
    for i, sensor in enumerate(humidity_sensors):
        if i < len(humidity_zones) and humidity_zones[i] == zone_id:
            zone_humidity_sensors.append(sensor)
    
    # Use existing air quality calculation function with filtered sensors
    return _worst_air_quality_index(
        hass,
        zone_co2_sensors,
        zone_voc_sensors,
        zone_pm_sensors,
        zone_humidity_sensors,
        thresholds_by_dc,
        previous_values,
        humidity_avg_tracker,
    )

def format_zone_command(id_from: str, id_to: str, rate: int, sensor_id: int = 255) -> str:
    """
    Convert rate to ramses_cc command format for a specific zone.
    
    Args:
        id_from: Source device ID (e.g., "29:162275")
        id_to: Target device ID (e.g., "32:146231") 
        rate: Fan power rate (0-255)
        sensor_id: Sensor identifier (0-255, defaults to 255)
        
    Returns:
        Formatted command string like " I --- 29:162275 32:146231 --:------ 31E0 008 00000A0001FFAA00"
    """
    # Convert rate and sensor_id to hex (2 digits, uppercase)
    rate_hex = f"{rate:02X}"
    sensor_id_hex = f"{sensor_id:02X}"
    
    # Build the command string with sensor_id in the appropriate position
    command = f" I --- {id_from} {id_to} --:------ 31E0 008 0000{rate_hex}000100{sensor_id_hex}00"
    
    return command

async def send_zone_commands_with_delay(hass: HomeAssistant, zone_configs: dict, zone_rates: dict, remote_entity_id: str, entry_data: dict) -> None:
    """
    Send zone commands with delays between them in a non-blocking way.
    
    Args:
        hass: Home Assistant instance
        zone_configs: Zone configuration dictionary
        zone_rates: Dictionary of zone rates to send
        remote_entity_id: Remote entity ID for sending commands
        entry_data: Entry data containing added commands
    """
    import asyncio
    
    added_commands = entry_data.get("added_commands", set())
    
    async def send_single_zone_command(zone_id: int, delay_seconds: float = 0):
        """Send command to a single zone with optional delay."""
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
            
        zone_config = zone_configs.get(zone_id)
        zone_rate = zone_rates.get(zone_id)
        
        if not zone_config or zone_rate is None:
            return
            
        id_from = zone_config.get(CONF_ZONE_ID_FROM)
        id_to = zone_config.get(CONF_ZONE_ID_TO)
        sensor_id = zone_config.get(CONF_ZONE_SENSOR_ID, 256 - zone_id)  # Default to 255 for zone 1, 254 for zone 2, etc.
        
        if not id_from or not id_to:
            _LOGGER.warning(f"Zone {zone_id} missing ID configuration, skipping command")
            return
            
        # Create unique command name for this zone and rate
        command_name = f"zone_{zone_id}_rate_{zone_rate}"
        
        # Check if this command has been added before
        if command_name not in added_commands:
            # Need to add the command first
            formatted_command = format_zone_command(id_from, id_to, zone_rate, sensor_id)
            _LOGGER.info(f"Adding new command '{command_name}': {formatted_command}")
            
            try:
                # Use asyncio.create_task to completely detach the add_command call
                # with timeout protection to prevent hanging
                async def add_command_with_timeout():
                    try:
                        await asyncio.wait_for(
                            hass.services.async_call(
                                "ramses_cc", "add_command",
                                {
                                    "entity_id": remote_entity_id,
                                    "command": command_name,
                                    "packet_string": formatted_command
                                },
                                blocking=False
                            ),
                            timeout=5.0  # 5 second timeout for add_command
                        )
                    except asyncio.TimeoutError:
                        _LOGGER.warning(f"Timeout adding command '{command_name}', continuing anyway")
                    except Exception as e:
                        _LOGGER.warning(f"Error adding command '{command_name}': {e}")
                
                add_task = hass.async_create_task(
                    add_command_with_timeout(),
                    name=f"ramses_add_{command_name}"
                )
                
                # Mark this command as added immediately (assume it will succeed)
                added_commands.add(command_name)
                entry_data["added_commands"] = added_commands
                _LOGGER.info(f"Successfully initiated adding command '{command_name}'")
                
                # Give ramses_cc a moment to process the add_command
                await asyncio.sleep(0.5)
                
            except Exception as e:
                _LOGGER.error(f"Failed to add command '{command_name}': {e}")
                return
        
        # Send the command using the added command name
        _LOGGER.debug(f"Sending command '{command_name}' to zone {zone_id} (rate: {zone_rate})")
        
        try:
            # Use asyncio.create_task to completely detach the service call
            # This ensures our PID controller never gets blocked by ramses_cc timeouts
            async def send_command_with_timeout():
                try:
                    await asyncio.wait_for(
                        hass.services.async_call(
                            "ramses_cc", "send_command",
                            {
                                "entity_id": remote_entity_id,
                                "command": command_name
                            },
                            blocking=False
                        ),
                        timeout=10.0  # 10 second timeout for send_command
                    )
                except asyncio.TimeoutError:
                    _LOGGER.warning(f"Timeout sending command '{command_name}' to zone {zone_id}")
                except Exception as e:
                    _LOGGER.warning(f"Error sending command '{command_name}' to zone {zone_id}: {e}")
            
            hass.async_create_task(
                send_command_with_timeout(),
                name=f"ramses_send_{command_name}"
            )
            _LOGGER.debug(f"Successfully initiated command to zone {zone_id}")
        except Exception as e:
            _LOGGER.error(f"Failed to send command '{command_name}' to zone {zone_id}: {e}")
    
    # Create tasks to send commands with staggered delays
    tasks = []
    delay_between_commands = 1.0  # 1 second between commands (reduced since non-blocking)
    
    for i, zone_id in enumerate(sorted(zone_rates.keys())):
        delay = i * delay_between_commands
        task = hass.async_create_task(
            send_single_zone_command(zone_id, delay),
            name=f"send_zone_{zone_id}_command"
        )
        tasks.append(task)
    
    # Start all tasks (they will run concurrently with their delays)
    if tasks:
        _LOGGER.debug(f"Starting {len(tasks)} zone command tasks with {delay_between_commands}s delays")
        # Don't await here - let them run in the background
        # The tasks will complete on their own schedule

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up PID Ventilation Control from a config entry."""
    # Merge options over data so runtime changes take immediate effect
    cfg = {**entry.data, **entry.options}
    # Get configuration from merged config
    co2_sensors = cfg.get(CONF_CO2_SENSORS, [])
    voc_sensors = cfg.get(CONF_VOC_SENSORS, [])
    pm_sensors = cfg.get(CONF_PM_SENSORS, [])
    humidity_sensors = cfg.get(CONF_HUMIDITY_SENSORS, [])
    
    # Get zone assignments
    co2_zones = cfg.get(CONF_CO2_SENSOR_ZONES, [])
    voc_zones = cfg.get(CONF_VOC_SENSOR_ZONES, [])
    pm_zones = cfg.get(CONF_PM_SENSOR_ZONES, [])
    humidity_zones = cfg.get(CONF_HUMIDITY_SENSOR_ZONES, [])
    
    # Get zone configurations and determine number of zones
    zone_configs = cfg.get(CONF_ZONE_CONFIGS, {})
    
    # Backward compatibility: if no zone configs but has old fan settings, create default zone config
    if not zone_configs:
        min_fan_output = cfg.get(CONF_MIN_FAN_OUTPUT, 0)
        max_fan_output = cfg.get(CONF_MAX_FAN_OUTPUT, 255)
        remote_device_id = cfg.get(CONF_REMOTE_DEVICE)
        fan_device_id = cfg.get(CONF_FAN_DEVICE)
        
        # Try to extract serial numbers from devices for backward compatibility
        remote_serial = None
        fan_serial = None
        
        if remote_device_id:
            try:
                device_registry = dr.async_get(hass)
                remote_device = device_registry.async_get(remote_device_id)
                if remote_device:
                    for domain, identifier in remote_device.identifiers:
                        if isinstance(identifier, str) and len(identifier) > 6:
                            remote_serial = identifier
                            break
            except Exception:
                pass
        
        if fan_device_id:
            try:
                device_registry = dr.async_get(hass)
                fan_device = device_registry.async_get(fan_device_id)
                if fan_device:
                    for domain, identifier in fan_device.identifiers:
                        if isinstance(identifier, str) and len(identifier) > 6:
                            fan_serial = identifier
                            break
            except Exception:
                pass
        
        # Create default zone config if we have the necessary data
        if remote_serial and fan_serial:
            zone_configs = {
                1: {
                    CONF_ZONE_ID_FROM: remote_serial,
                    CONF_ZONE_ID_TO: fan_serial,
                    CONF_ZONE_SENSOR_ID: 255,  # Default sensor ID for zone 1
                    CONF_ZONE_MIN_FAN: min_fan_output,
                    CONF_ZONE_MAX_FAN: max_fan_output,
                }
            }
            _LOGGER.info(f"Created backward compatibility zone config: {zone_configs[1]}")
    
    num_zones = len(zone_configs) if zone_configs else 1
    
    # Get valve devices for backward compatibility
    valve_devices = cfg.get(CONF_VALVE_DEVICES, [])
    
    setpoint = cfg[CONF_SETPOINT]
    kp_config = cfg[CONF_KP]
    ki_times = cfg[CONF_KI_TIMES]
    update_interval = cfg[CONF_UPDATE_INTERVAL]
    remote_device_id = cfg[CONF_REMOTE_DEVICE]
    fan_device_id = cfg.get(CONF_FAN_DEVICE)
    
    # Get remote device and extract entity_id and serial number
    remote_entity_id = None
    remote_serial_number = None
    if remote_device_id:
        try:
            device_registry = dr.async_get(hass)
            remote_device = device_registry.async_get(remote_device_id)
            if remote_device:
                # Extract serial number from device identifiers
                for domain, identifier in remote_device.identifiers:
                    if isinstance(identifier, str) and len(identifier) > 6:
                        remote_serial_number = identifier
                        break
                
                # Find remote entity from this device
                entity_registry = er.async_get(hass)
                entities = er.async_entries_for_device(entity_registry, remote_device_id)
                
                # Look for a remote entity
                for entity in entities:
                    if entity.domain == "remote":
                        remote_entity_id = entity.entity_id
                        break
                        
                _LOGGER.info(f"Remote device found: {remote_device.name}, Entity: {remote_entity_id}, Serial: {remote_serial_number}")
            else:
                _LOGGER.warning(f"Remote device with ID {remote_device_id} not found")
        except Exception as e:
            _LOGGER.error(f"Error accessing remote device {remote_device_id}: {e}", exc_info=True)
            return False
    
    # Get fan device and extract serial number if device is selected
    fan_serial_number = None
    if fan_device_id:
        try:
            device_registry = dr.async_get(hass)
            fan_device = device_registry.async_get(fan_device_id)
            if fan_device:
                # Extract serial number from device identifiers
                for domain, identifier in fan_device.identifiers:
                    if isinstance(identifier, str) and len(identifier) > 6:
                        # Assume the identifier is or contains the serial number
                        fan_serial_number = identifier
                        break
                _LOGGER.info(f"Fan device found: {fan_device.name}, Serial: {fan_serial_number}")
            else:
                _LOGGER.warning(f"Fan device with ID {fan_device_id} not found")
        except Exception as e:
            _LOGGER.error(f"Error accessing fan device {fan_device_id}: {e}", exc_info=True)
    
    # Indices (use defaults if not provided)
    co2_index = cfg.get(CONF_CO2_INDEX, DEFAULT_CO2_INDEX)
    voc_index = cfg.get(CONF_VOC_INDEX, DEFAULT_VOC_INDEX)
    voc_ppm_index = cfg.get(CONF_VOC_PPM_INDEX, DEFAULT_VOC_PPM_INDEX)
    pm_1_0_index = cfg.get(CONF_PM_1_0_INDEX, DEFAULT_PM_1_0_INDEX)
    pm_2_5_index = cfg.get(CONF_PM_2_5_INDEX, DEFAULT_PM_2_5_INDEX)
    pm_10_index = cfg.get(CONF_PM_10_INDEX, DEFAULT_PM_10_INDEX)
    humidity_index = cfg.get(CONF_HUMIDITY_INDEX, DEFAULT_HUMIDITY_INDEX)

    thresholds_by_dc = {
        "carbon_dioxide": co2_index,
        "volatile_organic_compounds": voc_index,
        "volatile_organic_compounds_parts": voc_ppm_index,
        "pm1": pm_1_0_index,
        "pm25": pm_2_5_index,
        "pm10": pm_10_index,
        "humidity": humidity_index,
    }
    
    _LOGGER.info(
        f"Ventilation setup - {num_zones} zones, CO2: {len(co2_sensors)}, VOC: {len(voc_sensors)}, PM: {len(pm_sensors)}, Humidity: {len(humidity_sensors)}, Setpoint: {setpoint}"
    )
    _LOGGER.info(f"Zone configs: {list(zone_configs.keys())}, Kp: {kp_config}, Update interval: {update_interval}s")
  
    # Calculate the overall fan range for PID calculations (use widest range from all zones)
    if zone_configs:
        all_min_rates = [zone_configs[zone_id].get(CONF_ZONE_MIN_FAN, 0) for zone_id in zone_configs]
        all_max_rates = [zone_configs[zone_id].get(CONF_ZONE_MAX_FAN, 255) for zone_id in zone_configs]
        global_min_rate = min(all_min_rates)
        global_max_rate = max(all_max_rates)
    else:
        # Fallback for single zone without configuration
        global_min_rate = 0
        global_max_rate = 255
    
    # Fan discrete speeds for PID calculation
    fan_speeds = np.arange(global_min_rate, global_max_rate + 1e-6, 1)

    # PID coefficients
    fan_diff = fan_speeds.max() - fan_speeds.min() if len(fan_speeds) > 1 else 255
    Kp = kp_config
    Ki = [fan_diff / ki_time for ki_time in ki_times]
    
    _LOGGER.info(f"Calculated Ki values: {Ki}")
    #Kd = 0.005

    # Initialize separate PID states for each zone
    zone_pid_states = {}
    for zone_id in range(1, num_zones + 1):
        zone_pid_states[zone_id] = {
            "integral": 0.0,
            "prev_error": 0.0,
        }

    # Previous values cache per entity
    previous_values: dict[str, float] = {}
    
    # Initialize 24-hour humidity moving average tracker
    humidity_avg_tracker = HumidityMovingAverage()

    async def pid_control(now):
        nonlocal zone_pid_states, previous_values
        # Check if controller is enabled
        domain_data = hass.data.get(DOMAIN, {})
        entry_data = domain_data.get(entry.entry_id, {}) if isinstance(domain_data, dict) else {}
        controller_enabled = entry_data.get("enabled", True)
            
        # Get humidity tracker and zone PID states from stored data
        humidity_avg_tracker = entry_data.get("humidity_avg_tracker", HumidityMovingAverage())
        stored_zone_pid_states = entry_data.get("zone_pid_states", {})
        
        # Update zone_pid_states with stored values if available
        for zone_id in zone_pid_states:
            if zone_id in stored_zone_pid_states:
                zone_pid_states[zone_id] = stored_zone_pid_states[zone_id]
        
        # Get current setpoint (may have been updated via number entity)
        runtime_setpoint = entry_data.get("current_setpoint")
        if runtime_setpoint is not None:
            current_setpoint = runtime_setpoint
        else:
            current_cfg = {**entry.data, **entry.options}
            current_setpoint = current_cfg.get(CONF_SETPOINT, setpoint)
         
        # Calculate time difference since last execution
        current_time = datetime.now()
        last_execution_time = entry_data.get("last_execution_time")
        if last_execution_time is not None:
            time_diff = (current_time - last_execution_time).total_seconds()
            _LOGGER.info(f"Time since last execution: {time_diff:.2f} seconds")
        else:
            _LOGGER.info("First execution - no previous time to compare")
            time_diff = float(update_interval)  # Use configured interval for first run
        
        # Update last execution time in entry data
        entry_data["last_execution_time"] = current_time

        # Calculate AQI per zone and find worst overall
        zone_aqi_data = {}
        worst_zone_aqi = 0.0
        worst_entity = ""
        worst_value = 0.0
        worst_dc = ""
        worst_zone = 1
        
        for zone_id in range(1, num_zones + 1):
            zone_aqi, zone_worst_entity, zone_worst_value, zone_worst_dc = _calculate_zone_air_quality(
                hass,
                zone_id,
                co2_sensors,
                voc_sensors,
                pm_sensors,
                humidity_sensors,
                co2_zones,
                voc_zones,
                pm_zones,
                humidity_zones,
                thresholds_by_dc,
                previous_values,
                humidity_avg_tracker,
            )
            
            zone_aqi_data[zone_id] = zone_aqi
            
            # Track worst zone for overall fan control
            if zone_aqi > worst_zone_aqi:
                worst_zone_aqi = zone_aqi
                worst_entity = zone_worst_entity
                worst_value = zone_worst_value
                worst_dc = zone_worst_dc
                worst_zone = zone_id
        
        # Use worst zone AQI for fan control
        air_quality_index = worst_zone_aqi

        # Update shared state and notify subscribers (sensor)
        try:
            ed = hass.data.get(DOMAIN, {}).get(entry.entry_id)
            if isinstance(ed, dict):
                ed["aqi"] = air_quality_index  # Overall worst AQI
                ed["zone_aqi"] = zone_aqi_data  # Per-zone AQI data
                ed["worst_zone"] = worst_zone
                ed["worst_entity"] = worst_entity
                ed["worst_value"] = worst_value
                ed["worst_dc"] = worst_dc
                ed["humidity_avg_tracker"] = humidity_avg_tracker
            async_dispatcher_send(
                hass,
                f"{SIGNAL_AQI_UPDATED}_{entry.entry_id}",
                {
                    "aqi": air_quality_index,
                    "zone_aqi": zone_aqi_data,
                    "worst_zone": worst_zone,
                    "worst_entity": worst_entity,
                    "worst_value": worst_value,
                    "worst_dc": worst_dc,
                },
            )
        except Exception:  # best-effort notification
            pass

        # Initialize zone rates
        zone_rates = {}
        max_zone_rate = 0
        
        # Check if PID control is enabled
        if not controller_enabled:
            # Controller is disabled - set all zones to rate 0 but continue monitoring
            _LOGGER.debug("Controller disabled - setting all zones to rate 0")
            
            # Reset time baseline to avoid integral spike when re-enabled
            entry_data["last_execution_time"] = datetime.now()
            
            # Set all zones to zero rate
            if zone_configs:
                for zone_id in zone_configs.keys():
                    zone_rates[zone_id] = 0
            
            # Update rate sensors with zero values but skip PID calculations and commands
            max_zone_rate = 0
        else:
            # Controller is enabled - run PID calculations and send commands
            _LOGGER.debug("Controller enabled - running PID calculations")
            
            # Check if we have zone configurations
            if not zone_configs:
                _LOGGER.warning("No zone configurations found, cannot send fan commands")
                # Set empty rates for sensors
                for zone_id in range(1, num_zones + 1):
                    zone_rates[zone_id] = 0
                max_zone_rate = 0
            else:
                # Run separate PID controllers for each zone
                for zone_id, zone_config in zone_configs.items():
                    # Get zone-specific parameters
                    id_from = zone_config.get(CONF_ZONE_ID_FROM)
                    id_to = zone_config.get(CONF_ZONE_ID_TO)
                    sensor_id = zone_config.get(CONF_ZONE_SENSOR_ID, 256 - int(zone_id))  # Default to 255 for zone 1, 254 for zone 2, etc.
                    zone_min = zone_config.get(CONF_ZONE_MIN_FAN, 0)
                    zone_max = zone_config.get(CONF_ZONE_MAX_FAN, 255)
                    
                    if not id_from or not id_to:
                        _LOGGER.warning(f"Zone {zone_id} missing ID configuration, skipping")
                        continue
                    
                    # Get zone-specific AQI
                    zone_aqi = zone_aqi_data.get(zone_id, 0.0)
                    
                    # Get PID state for this zone
                    pid_state = zone_pid_states.get(zone_id, {"integral": 0.0, "prev_error": 0.0})
                    
                    # Calculate zone-specific PID output
                    idx = min(int(round(zone_aqi)), len(Ki) - 1)
                    Ki_index_based = Ki[idx]
                    error = zone_aqi - current_setpoint
                    pid_state["integral"] += error * Ki_index_based * time_diff
                    pid_state["prev_error"] = error
                    
                    # Zone-specific fan speeds range
                    zone_fan_speeds = np.arange(zone_min, zone_max + 1e-6, 1)
                    
                    # PID output (desired removal rate) for this zone
                    pid_output = Kp * error + pid_state["integral"] + zone_fan_speeds.min()
                    
                    # Map to discrete fan speed and handle integral windup for this zone
                    if pid_output < zone_fan_speeds[0]:
                        zone_rate = int(zone_fan_speeds[0])
                        pid_state["integral"] -= error * Ki_index_based * time_diff  # Remove the integral contribution that caused windup
                    elif pid_output > zone_fan_speeds[-1]:
                        zone_rate = int(zone_fan_speeds[-1])
                        pid_state["integral"] -= error * Ki_index_based * time_diff  # Remove the integral contribution that caused windup
                    else:
                        # Round to nearest integer within allowed range
                        zone_rate = int(round(float(np.clip(pid_output, zone_fan_speeds[0], zone_fan_speeds[-1]))))
                    
                    # Store the updated PID state
                    zone_pid_states[zone_id] = pid_state
                    zone_rates[zone_id] = zone_rate
                    
                    # Track maximum rate for overall sensor
                    max_zone_rate = max(max_zone_rate, zone_rate)
                    
                    _LOGGER.debug(f"Zone {zone_id} PID: AQI={zone_aqi:.2f}, Setpoint={current_setpoint:.2f}, Error={error:.2f}, Rate={zone_rate}, Integral={pid_state['integral']:.2f}")
                
                # Send commands to each zone with delays between them (non-blocking) - only when enabled
                try:
                    await send_zone_commands_with_delay(hass, zone_configs, zone_rates, remote_entity_id, entry_data)
                except Exception as e:
                    _LOGGER.warning(f"Error in zone command transmission (continuing PID operation): {e}")
                    # Don't let ramses_cc errors stop the PID controller

        # Compute and publish rate percentage (use max zone rate for overall percentage)
        global_rate = max_zone_rate
        try:
            span = global_max_rate - global_min_rate
            pct = 0.0 if span <= 0 else round((global_rate - global_min_rate) * 100.0 / span, 2)
            ed = hass.data.get(DOMAIN, {}).get(entry.entry_id)
            if isinstance(ed, dict):
                ed["rate_pct"] = pct
                ed["zone_rates"] = zone_rates  # Store individual zone rates
                ed["zone_pid_states"] = zone_pid_states  # Store PID states
            async_dispatcher_send(
                hass,
                f"{SIGNAL_RATE_UPDATED}_{entry.entry_id}",
                {"rate_pct": pct, "rate": global_rate, "zone_rates": zone_rates},
            )
        except Exception:
            pass

        # Log zone AQI and rate information
        zone_aqi_str = ", ".join([f"Zone {zone}: AQI={aqi:.2f}" for zone, aqi in zone_aqi_data.items()])
        zone_rates_str = ", ".join([f"Zone {zone}: {rate}" for zone, rate in zone_rates.items()])
        zone_pid_str = ", ".join([f"Zone {zone}: I={zone_pid_states.get(zone, {}).get('integral', 0):.2f}" for zone in zone_rates.keys()])
        _LOGGER.info(
            f"Dual PID: {zone_aqi_str}, Worst=Zone {worst_zone} ({worst_entity}, dc={worst_dc}, val={worst_value}), Setpoint={current_setpoint:.2f}, Rates({zone_rates_str}), Integrals({zone_pid_str})"
        )

    # Store the remove callback and runtime state in hass.data for cleanup and control
    remove_listener = async_track_time_interval(hass, pid_control, timedelta(seconds=update_interval))
    
    # Add options update listener with delayed reload to prevent hanging
    entry.add_update_listener(async_delayed_reload_entry)
    
    if DOMAIN not in hass.data:
        hass.data[DOMAIN] = {}
    hass.data[DOMAIN][entry.entry_id] = {
        "enabled": True,
        "remove_listener": remove_listener,
        "aqi": 0.0,
        "zone_rates": {},  # Track individual zone rates
        "worst_entity": None,
        "worst_value": None,
        "worst_dc": None,
        "rate_pct": 0.0,
        "remote_device_id": remote_device_id,
        "remote_entity_id": remote_entity_id,
        "remote_serial_number": remote_serial_number,
        "zone_configs": zone_configs,  # Store zone configurations
        "zone_pid_states": zone_pid_states,  # Store PID states for each zone
        "added_commands": set(),  # Track which rate commands have been added
        "last_execution_time": None,  # Instance-specific last execution time
        "current_setpoint": setpoint,  # Initialize with config setpoint
        "humidity_avg_tracker": humidity_avg_tracker,  # Store humidity moving average tracker
    }

    # Ensure switch platform is set up after state exists
    try:
        await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
        _LOGGER.info(f"Successfully set up platforms: {PLATFORMS}")
    except Exception as e:
        _LOGGER.error(f"Error setting up platforms: {e}", exc_info=True)
        # Clean up partial setup
        await async_unload_entry(hass, entry)
        return False
 
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry with improved error handling."""
    _LOGGER.debug(f"Starting unload for config entry {entry.entry_id}")
    
    unload_ok = True
    try:
        # Clean up timer and data first to stop the PID controller
        _LOGGER.debug("Cleaning up data...")
        try:
            domain_data = hass.data.get(DOMAIN, {})
            entry_data = domain_data.get(entry.entry_id, {})
            
            if isinstance(entry_data, dict):
                remove_listener = entry_data.get("remove_listener")
                if remove_listener:
                    try:
                        remove_listener()
                        _LOGGER.debug("Successfully removed timer listener")
                    except Exception as e:
                        _LOGGER.warning(f"Error removing timer listener: {e}")

            # Remove entry data but keep domain data for other entries
            domain_data.pop(entry.entry_id, None)
            
            # Clean up domain data if empty
            if not domain_data:
                hass.data.pop(DOMAIN, None)
                
        except Exception as e:
            _LOGGER.error(f"Error cleaning up data: {e}")
            unload_ok = False

        # Unload platforms with individual error handling
        _LOGGER.debug("Unloading platforms...")
        try:
            unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
            if not unload_ok:
                _LOGGER.warning(f"Some platforms failed to unload for entry {entry.entry_id}")
        except ValueError as ve:
            if "never loaded" in str(ve):
                _LOGGER.debug(f"Platforms were not loaded, skipping unload: {ve}")
                unload_ok = True  # This is OK - nothing to unload
            else:
                _LOGGER.error(f"ValueError unloading platforms: {ve}")
                unload_ok = False
        except Exception as e:
            _LOGGER.error(f"Error unloading platforms: {e}")
            unload_ok = False
        
        _LOGGER.debug(f"Unload completed for config entry {entry.entry_id} - Success: {unload_ok}")
        return unload_ok
        
    except Exception as e:
        _LOGGER.error(f"Unexpected error during unload of {entry.entry_id}: {e}", exc_info=True)
        return False


async def async_delayed_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Schedule a delayed reload to prevent hanging during options flow."""
    _LOGGER.info(f"Scheduling delayed reload for config entry {entry.entry_id}")
    
    # Use call_later to delay the reload and prevent blocking the options flow
    def schedule_reload():
        hass.async_create_task(async_reload_entry_safe(hass, entry))
    
    # Schedule reload after 2 seconds to allow options flow to complete
    hass.loop.call_later(2.0, schedule_reload)


async def async_reload_entry_safe(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Safe reload that won't hang the UI."""
    _LOGGER.info(f"Starting safe reload for config entry {entry.entry_id}")
    
    # Add a flag to prevent reload loops
    if hasattr(entry, "_reloading"):
        _LOGGER.warning("Already reloading entry, skipping to prevent loop")
        return
    
    try:
        entry._reloading = True
        
        # Preserve runtime setpoint during reload
        domain_data = hass.data.get(DOMAIN, {})
        entry_data = domain_data.get(entry.entry_id, {})
        preserved_setpoint = entry_data.get("current_setpoint")
        
        _LOGGER.debug(f"Safe reload - Options: {len(entry.options)} items, Entry data: {len(entry.data)} items")
        
        # Unload first
        try:
            _LOGGER.debug("Unloading platforms safely...")
            unload_success = await async_unload_entry(hass, entry)
            if unload_success:
                _LOGGER.debug("Platforms unloaded successfully")
            else:
                _LOGGER.warning("Some platforms failed to unload, continuing with setup")
        except Exception as unload_error:
            _LOGGER.warning(f"Error during safe unload: {unload_error}")
        
        # Small delay to ensure cleanup completes
        import asyncio
        await asyncio.sleep(0.5)
        
        # Setup again
        try:
            _LOGGER.debug("Setting up platforms safely...")
            setup_success = await async_setup_entry(hass, entry)
            
            if not setup_success:
                _LOGGER.error("Safe setup returned False")
                return
                
            _LOGGER.debug("Platforms set up successfully")
        except Exception as setup_error:
            _LOGGER.error(f"Error during safe setup: {setup_error}")
            return
        
        # Restore runtime setpoint after reload
        if preserved_setpoint is not None:
            domain_data = hass.data.get(DOMAIN, {})
            entry_data = domain_data.get(entry.entry_id, {})
            if isinstance(entry_data, dict):
                entry_data["current_setpoint"] = preserved_setpoint
                _LOGGER.debug(f"Restored runtime setpoint: {preserved_setpoint}")
        
        _LOGGER.info(f"Successfully completed safe reload for config entry {entry.entry_id}")
        
    except Exception as e:
        _LOGGER.error(f"Error in safe reload for {entry.entry_id}: {e}", exc_info=True)
    finally:
        # Always remove reload flag
        if hasattr(entry, "_reloading"):
            delattr(entry, "_reloading")


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload config entry with improved error handling."""
    _LOGGER.info(f"Reloading config entry {entry.entry_id}")
    
    # Add a flag to prevent reload loops and add timeout
    if hasattr(entry, "_reloading"):
        _LOGGER.warning("Already reloading entry, skipping to prevent loop")
        return
    
    try:
        entry._reloading = True
        
        # Preserve runtime setpoint during reload
        domain_data = hass.data.get(DOMAIN, {})
        entry_data = domain_data.get(entry.entry_id, {})
        preserved_setpoint = entry_data.get("current_setpoint")
        
        _LOGGER.debug(f"Starting reload - Options: {len(entry.options)} items, Entry data: {len(entry.data)} items")
        
        # First, try to unload cleanly with timeout
        try:
            _LOGGER.debug("Unloading platforms...")
            await hass.async_create_task(
                async_unload_entry(hass, entry),
                name=f"unload_{entry.entry_id}"
            )
            _LOGGER.debug("Platforms unloaded successfully")
        except Exception as unload_error:
            _LOGGER.warning(f"Error during unload: {unload_error}")
            # Continue with setup even if unload failed
        
        # Then setup with timeout  
        try:
            _LOGGER.debug("Setting up platforms...")
            setup_success = await hass.async_create_task(
                async_setup_entry(hass, entry),
                name=f"setup_{entry.entry_id}"
            )
            
            if not setup_success:
                raise Exception("Setup returned False")
                
            _LOGGER.debug("Platforms set up successfully")
        except Exception as setup_error:
            _LOGGER.error(f"Error during setup: {setup_error}")
            raise
        
        # Restore runtime setpoint after reload
        if preserved_setpoint is not None:
            domain_data = hass.data.get(DOMAIN, {})
            entry_data = domain_data.get(entry.entry_id, {})
            if isinstance(entry_data, dict):
                entry_data["current_setpoint"] = preserved_setpoint
                _LOGGER.debug(f"Restored runtime setpoint: {preserved_setpoint}")
        
        _LOGGER.info(f"Successfully reloaded config entry {entry.entry_id}")
        
    except Exception as e:
        _LOGGER.error(f"Error reloading config entry {entry.entry_id}: {e}", exc_info=True)
        
        # Try to restore integration in a working state
        try:
            _LOGGER.warning("Attempting recovery setup with original configuration")
            await async_setup_entry(hass, entry)
            _LOGGER.info("Recovery setup completed")
        except Exception as recovery_error:
            _LOGGER.error(f"Recovery setup failed: {recovery_error}", exc_info=True)
            # Don't raise - let Home Assistant handle the failed integration
    finally:
        # Always remove reload flag
        if hasattr(entry, "_reloading"):
            delattr(entry, "_reloading")


async def async_setup(hass, config):
    """Set up the ventilation component from YAML (deprecated)."""
    # This is kept for backward compatibility but should not be used
    # All new installations should use the config flow
    return True
