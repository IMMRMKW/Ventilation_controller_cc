from datetime import timedelta, datetime
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.const import Platform
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
import homeassistant.helpers.config_validation as cv
import voluptuous as vol
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
        sensor_id = zone_config.get(CONF_ZONE_SENSOR_ID, 256 - int(zone_id))  # Default to 255 for zone 1, 254 for zone 2, etc.
        
        if not id_from or not id_to:
            _LOGGER.warning(f"Zone {zone_id} missing ID configuration, skipping command")
            return
            
        # Create unique command name that includes sensor_id and device IDs to handle config changes
        # This ensures that when sensor ID or devices change, new commands are created
        id_from_short = id_from.replace(":", "")[-6:]  # Last 6 chars of id_from
        id_to_short = id_to.replace(":", "")[-6:]      # Last 6 chars of id_to
        command_name = f"zone_{zone_id}_rate_{zone_rate}_sensor_{sensor_id}_{id_from_short}_{id_to_short}"
        
        # Clear any old commands for this zone that might have different sensor/device config
        # This handles cases where sensor ID or device IDs changed
        old_commands_to_remove = []
        for existing_cmd in added_commands:
            if existing_cmd.startswith(f"zone_{zone_id}_rate_{zone_rate}_") and existing_cmd != command_name:
                old_commands_to_remove.append(existing_cmd)
        
        # Remove old commands from tracking (ramses_cc will keep them, but we won't reuse them)
        for old_cmd in old_commands_to_remove:
            added_commands.discard(old_cmd)
            _LOGGER.info(f"Removed old command '{old_cmd}' due to configuration change")
        
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
            async def send_command_with_retry():
                try:
                    # First attempt to send the command
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
                    _LOGGER.debug(f"Successfully sent command '{command_name}' to zone {zone_id}")
                    
                except Exception as send_error:
                    _LOGGER.warning(f"Send command '{command_name}' failed: {send_error}. Attempting to re-add command and retry...")
                    
                    # Command likely doesn't exist (ramses_cc restarted), re-add it and try again
                    try:
                        # Re-add the command
                        formatted_command = format_zone_command(id_from, id_to, zone_rate, sensor_id)
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
                        _LOGGER.info(f"Re-added command '{command_name}' after send failure")
                        
                        # Small delay to let ramses_cc process the add
                        await asyncio.sleep(0.2)
                        
                        # Retry sending the command
                        await asyncio.wait_for(
                            hass.services.async_call(
                                "ramses_cc", "send_command",
                                {
                                    "entity_id": remote_entity_id,
                                    "command": command_name
                                },
                                blocking=False
                            ),
                            timeout=10.0  # 10 second timeout for retry
                        )
                        _LOGGER.info(f"Successfully sent command '{command_name}' to zone {zone_id} after re-add")
                        
                    except Exception as retry_error:
                        _LOGGER.error(f"Failed to re-add or retry command '{command_name}' for zone {zone_id}: {retry_error}")
                        # Remove from added_commands so it gets re-added next time
                        added_commands.discard(command_name)
            
            hass.async_create_task(
                send_command_with_retry(),
                name=f"ramses_send_{command_name}"
            )
            _LOGGER.debug(f"Successfully initiated command to zone {zone_id}")
        except Exception as e:
            _LOGGER.error(f"Failed to initiate command '{command_name}' to zone {zone_id}: {e}")
    
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
    # Import the improved merge function to prevent sensor duplication bugs
    from .config_flow import merge_config_with_validation
    
    # Get properly merged and validated configuration
    cfg = merge_config_with_validation(entry.data, entry.options, allow_cleanup=False)
    
    # Get configuration from validated config
    co2_sensors = cfg.get(CONF_CO2_SENSORS, [])
    voc_sensors = cfg.get(CONF_VOC_SENSORS, [])
    pm_sensors = cfg.get(CONF_PM_SENSORS, [])
    humidity_sensors = cfg.get(CONF_HUMIDITY_SENSORS, [])
    
    # Get zone assignments
    co2_zones = cfg.get(CONF_CO2_SENSOR_ZONES, [])
    voc_zones = cfg.get(CONF_VOC_SENSOR_ZONES, [])
    pm_zones = cfg.get(CONF_PM_SENSOR_ZONES, [])
    humidity_zones = cfg.get(CONF_HUMIDITY_SENSOR_ZONES, [])
    
    # Get validated zone configurations 
    zone_configs = cfg.get(CONF_ZONE_CONFIGS, {})
    
    # Determine number of zones from validated config (now guaranteed consistent)
    num_zones = len(zone_configs) if zone_configs else int(cfg.get(CONF_NUM_ZONES, 1))
    
    _LOGGER.info(f"Setup entry: {num_zones} zones from validated config")
    
    # Process zone configs to ensure device serials are available
    # New configs store both device IDs and serials, old configs may only have serials
    processed_zone_configs = {}
    device_registry = dr.async_get(hass)
    
    for zone_id, zone_config in zone_configs.items():
        processed_config = zone_config.copy()
        
        # Check if we have device IDs that need serial extraction
        device_from = zone_config.get("device_from")
        device_to = zone_config.get("device_to")
        
        if device_from and not zone_config.get(CONF_ZONE_ID_FROM):
            # Extract serial from device_from if not already available
            try:
                device = device_registry.async_get(device_from)
                if device:
                    for domain, identifier in device.identifiers:
                        if isinstance(identifier, str) and len(identifier) > 6:
                            processed_config[CONF_ZONE_ID_FROM] = identifier
                            break
            except Exception as e:
                _LOGGER.warning(f"Failed to extract serial from device_from {device_from} for zone {zone_id}: {e}")
        
        if device_to and not zone_config.get(CONF_ZONE_ID_TO):
            # Extract serial from device_to if not already available
            try:
                device = device_registry.async_get(device_to)
                if device:
                    for domain, identifier in device.identifiers:
                        if isinstance(identifier, str) and len(identifier) > 6:
                            processed_config[CONF_ZONE_ID_TO] = identifier
                            break
            except Exception as e:
                _LOGGER.warning(f"Failed to extract serial from device_to {device_to} for zone {zone_id}: {e}")
        
        processed_zone_configs[zone_id] = processed_config
    
    zone_configs = processed_zone_configs
    
    # Note: Zone configurations must now be set up through the config flow
    # Legacy single fan device setup is no longer supported
    
    # Use validated num_zones from consistent config helper
    _LOGGER.debug(f"Using validated zone count: {num_zones}, zone_configs: {list(zone_configs.keys())}")
    
    # Get valve devices for backward compatibility
    valve_devices = cfg.get(CONF_VALVE_DEVICES, [])
    
    setpoint = cfg[CONF_SETPOINT]
    kp_config = cfg[CONF_KP]
    ki_times = cfg[CONF_KI_TIMES]
    update_interval = cfg[CONF_UPDATE_INTERVAL]
    remote_device_id = cfg[CONF_REMOTE_DEVICE]
    
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
    # Zone configs now store percentages (0-100), so we work in percentage space
    if zone_configs:
        all_min_rates_pct = [zone_configs[zone_id].get(CONF_ZONE_MIN_FAN, 0) for zone_id in zone_configs]
        all_max_rates_pct = [zone_configs[zone_id].get(CONF_ZONE_MAX_FAN, 100) for zone_id in zone_configs]
        global_min_rate_pct = min(all_min_rates_pct)
        global_max_rate_pct = max(all_max_rates_pct)
    else:
        # Fallback for single zone without configuration
        global_min_rate_pct = 0
        global_max_rate_pct = 100
    
    # Fan discrete speeds for PID calculation (now in percentage space)
    fan_speeds_pct = np.arange(global_min_rate_pct, global_max_rate_pct + 1e-6, 1)

    # PID coefficients (now based on percentage range)
    fan_diff_pct = fan_speeds_pct.max() - fan_speeds_pct.min() if len(fan_speeds_pct) > 1 else 100
    Kp = kp_config
    Ki = [fan_diff_pct / ki_time for ki_time in ki_times]
    
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
        
        # Get current configuration (includes any options updates)
        current_cfg = {**entry.data, **entry.options}
        
        # Reload zone configurations from current merged config to pick up any device changes
        from .config_flow import merge_config_with_validation
        current_merged_cfg = merge_config_with_validation(entry.data, entry.options, allow_cleanup=False)
        current_zone_configs = current_merged_cfg.get(CONF_ZONE_CONFIGS, {})
        
        # Get current setpoint (may have been updated via number entity)
        runtime_setpoint = entry_data.get("current_setpoint")
        if runtime_setpoint is not None:
            current_setpoint = runtime_setpoint
        else:
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
        
        # Use current number of zones from current config
        current_num_zones = len(current_zone_configs) if current_zone_configs else int(current_merged_cfg.get(CONF_NUM_ZONES, 1))
        
        for zone_id in range(1, current_num_zones + 1):
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
            
            # Set all zones to zero rate - ensure consistent zone coverage
            if zone_configs:
                for zone_id in zone_configs.keys():
                    zone_rates[zone_id] = 0
            else:
                # Fallback to ensure at least zone 1 exists for sensors
                for zone_id in range(1, num_zones + 1):
                    zone_rates[zone_id] = 0
            
            # Update rate sensors with zero values but skip PID calculations and commands
            max_zone_rate = 0
        else:
            # Controller is enabled - run PID calculations and send commands
            _LOGGER.debug("Controller enabled - running PID calculations")
            
            # Check if we have zone configurations
            if not current_zone_configs:
                _LOGGER.warning("No zone configurations found, cannot send fan commands")
                # Set empty rates for sensors
                for zone_id in range(1, current_num_zones + 1):
                    zone_rates[zone_id] = 0
                max_zone_rate = 0
            else:
                # Run separate PID controllers for each zone
                for zone_id, zone_config in current_zone_configs.items():
                    # Get zone-specific parameters
                    id_from = zone_config.get(CONF_ZONE_ID_FROM)
                    id_to = zone_config.get(CONF_ZONE_ID_TO)
                    sensor_id = zone_config.get(CONF_ZONE_SENSOR_ID, 256 - int(zone_id))  # Default to 255 for zone 1, 254 for zone 2, etc.
                    zone_min_pct = zone_config.get(CONF_ZONE_MIN_FAN, 0)  # Now in percentage
                    zone_max_pct = zone_config.get(CONF_ZONE_MAX_FAN, 100)  # Now in percentage
                    
                    if not id_from or not id_to:
                        _LOGGER.warning(f"Zone {zone_id} missing ID configuration, setting rate to 0")
                        zone_rates[zone_id] = 0
                        continue
                    
                    # Get zone-specific AQI
                    zone_aqi = zone_aqi_data.get(zone_id, 0.0)
                    
                    # Get PID state for this zone
                    pid_state = zone_pid_states.get(zone_id, {"integral": 0.0, "prev_error": 0.0})
                    
                    # Calculate zone-specific PID output
                    idx = min(int(round(zone_aqi)), len(Ki) - 1)
                    Ki_index_based = Ki[idx]
                    error = zone_aqi - current_setpoint
                    pid_state["integral"] += error * time_diff
                    pid_state["prev_error"] = error
                    
                    # Zone-specific fan speeds range (in percentage)
                    zone_fan_speeds_pct = np.arange(zone_min_pct, zone_max_pct + 1e-6, 1)
                    
                    # PID output (desired removal rate) for this zone (in percentage space)
                    pid_output_pct = Kp * error + Ki_index_based * pid_state["integral"] + zone_fan_speeds_pct.min()
                    
                    # Map to discrete fan speed and handle integral windup for this zone
                    if pid_output_pct < zone_fan_speeds_pct[0]:
                        zone_rate_pct = int(zone_fan_speeds_pct[0])
                        pid_state["integral"] -= error * time_diff  # Remove the integral contribution that caused windup
                    elif pid_output_pct > zone_fan_speeds_pct[-1]:
                        zone_rate_pct = int(zone_fan_speeds_pct[-1])
                        pid_state["integral"] -= error * time_diff  # Remove the integral contribution that caused windup
                    else:
                        # Round to nearest integer within allowed range
                        zone_rate_pct = int(round(float(np.clip(pid_output_pct, zone_fan_speeds_pct[0], zone_fan_speeds_pct[-1]))))
                    
                    # Convert percentage to 0-255 range only for command sending
                    zone_rate_255 = int(round((zone_rate_pct / 100.0) * 255))
                    
                    # Store the updated PID state
                    zone_pid_states[zone_id] = pid_state
                    
                    # Store both the percentage rate for sensors and the 0-255 rate for commands
                    zone_rates[zone_id] = zone_rate_255  # Commands use 0-255 range
                    
                    # Track maximum rate for overall sensor (using percentage)
                    max_zone_rate = max(max_zone_rate, zone_rate_pct)
                    
                    _LOGGER.debug(f"Zone {zone_id} PID: AQI={zone_aqi:.2f}, Setpoint={current_setpoint:.2f}, Error={error:.2f}, Rate={zone_rate_pct}% ({zone_rate_255}/255), Integral={pid_state['integral']:.2f}")
                
                # Send commands to each zone with delays between them (non-blocking) - only when enabled
                try:
                    await send_zone_commands_with_delay(hass, current_zone_configs, zone_rates, remote_entity_id, entry_data)
                except Exception as e:
                    _LOGGER.warning(f"Error in zone command transmission (continuing PID operation): {e}")
                    # Don't let ramses_cc errors stop the PID controller

        # Compute and publish rate percentage (max_zone_rate is already in percentage)
        global_rate_pct = max_zone_rate  # This is already the percentage value
        global_rate_255 = int(round((global_rate_pct / 100.0) * 255))  # Convert to 0-255 for compatibility
        try:
            # Since we're working in percentage space, the percentage is directly available
            pct = round(global_rate_pct, 2)
            
            # Ensure hass.data structure exists and update rate data
            if DOMAIN not in hass.data:
                hass.data[DOMAIN] = {}
            if entry.entry_id not in hass.data[DOMAIN]:
                hass.data[DOMAIN][entry.entry_id] = {}
                
            ed = hass.data[DOMAIN][entry.entry_id]
            ed["rate_pct"] = pct
            ed["zone_rates"] = zone_rates  # Store individual zone rates
            ed["zone_pid_states"] = zone_pid_states  # Store PID states
            
            # Send sensor update signal
            async_dispatcher_send(
                hass,
                f"{SIGNAL_RATE_UPDATED}_{entry.entry_id}",
                {"rate_pct": pct, "rate": global_rate_255, "zone_rates": zone_rates},
            )
            _LOGGER.debug(f"Updated rate sensor: {pct}% (rate: {global_rate_255}/255, zones: {zone_rates})")
            
        except Exception as e:
            _LOGGER.error(f"Error updating rate sensor: {e}", exc_info=True)
            # Ensure we still try to send a basic update even if percentage calculation fails
            try:
                async_dispatcher_send(
                    hass,
                    f"{SIGNAL_RATE_UPDATED}_{entry.entry_id}",
                    {"rate_pct": 0.0, "rate": global_rate_255, "zone_rates": zone_rates},
                )
            except Exception as e2:
                _LOGGER.error(f"Failed to send fallback rate update: {e2}")

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
    
    # Check if this entry already has data (e.g., from a reload) and preserve runtime setpoint
    existing_data = hass.data[DOMAIN].get(entry.entry_id, {})
    existing_runtime_setpoint = existing_data.get("current_setpoint") if isinstance(existing_data, dict) else None
    
    # Use existing runtime setpoint if available, otherwise fall back to config setpoint
    current_setpoint_value = existing_runtime_setpoint if existing_runtime_setpoint is not None else setpoint
    
    if existing_runtime_setpoint is not None:
        _LOGGER.debug(f"Preserving existing runtime setpoint: {existing_runtime_setpoint}")
    else:
        _LOGGER.debug(f"Initializing setpoint from config: {setpoint}")
    
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
        "current_setpoint": current_setpoint_value,  # Preserve runtime setpoint if available
        "humidity_avg_tracker": humidity_avg_tracker,  # Store humidity moving average tracker
    }

    # Register services for binding CO2 sensors to zones
    await _register_bind_services(hass, entry, zone_configs)

    # Clean up orphaned zone entities if number of zones was reduced
    await _cleanup_orphaned_zone_entities(hass, entry, num_zones)

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


async def _register_bind_services(hass: HomeAssistant, entry: ConfigEntry, zone_configs: dict):
    """Register bind_co2_n services for each configured zone."""
    
    async def handle_bind_co2_service(call: ServiceCall):
        """Handle bind_co2_n service calls."""
        service_name = call.service
        zone_number = service_name.replace("bind_co2_", "")
        
        try:
            zone_id = int(zone_number)  # Convert to integer, not string
            if zone_id not in zone_configs:
                _LOGGER.error(f"Zone {zone_number} not found in configuration. Available zones: {list(zone_configs.keys())}")
                return
                
            zone_config = zone_configs[zone_id]
            
            # Get device IDs from zone configuration
            device_id = zone_config.get(CONF_ZONE_ID_FROM)  # The device being bound (e.g., CO2 sensor)
            controller_id = zone_config.get(CONF_ZONE_ID_TO)  # The controller/target device
            
            if not device_id or not controller_id:
                _LOGGER.error(f"Zone {zone_number} missing ID configuration - id_from: {device_id}, id_to: {controller_id}")
                return
            
            # Get remote entity from the hass data
            domain_data = hass.data.get(DOMAIN, {})
            entry_data = domain_data.get(entry.entry_id, {})
            remote_entity_id = entry_data.get("remote_entity_id")
            
            if not remote_entity_id:
                _LOGGER.error(f"Remote entity not found for zone {zone_number} binding")
                return
            
            _LOGGER.info(f"Starting CO2 sensor binding for zone {zone_number}: {device_id} → {controller_id}")
            
            # Execute the binding sequence with zone-specific IDs
            await _execute_binding_sequence(hass, device_id, controller_id, remote_entity_id, zone_number)
            
        except Exception as e:
            _LOGGER.error(f"Error in bind_co2_{zone_number} service: {e}", exc_info=True)
    
    # Register a service for each configured zone
    for zone_id in zone_configs.keys():
        service_name = f"bind_co2_{zone_id}"
        
        # Check if service is already registered to avoid duplicates
        if not hass.services.has_service(DOMAIN, service_name):
            hass.services.async_register(
                DOMAIN,
                service_name,
                handle_bind_co2_service,
                schema=vol.Schema({}),  # No parameters needed
            )
            _LOGGER.info(f"Registered service: {DOMAIN}.{service_name}")


def _device_id_to_hex(device_id: str) -> str:
    """
    Convert device ID like '29:181233' to ramses hex format.
    
    Uses a mathematical approach based on reverse-engineered protocol:
    hex_value = device_type << 18 + device_number
    
    This gives each device type a 256K address space block.
    """
    parts = device_id.split(':')
    if len(parts) != 2:
        raise ValueError(f"Invalid device ID format: {device_id}")
    
    try:
        device_type = int(parts[0])
        device_num = int(parts[1])
    except ValueError:
        raise ValueError(f"Invalid device ID numbers: {device_id}")
    
    # Mathematical conversion based on ramses protocol analysis
    # Each device type gets a 256K block: type << 18
    base_offset = device_type << 18
    hex_value = base_offset + device_num
    
    return f"{hex_value:06X}"


def _create_device_offer_1fc9(device_id: str) -> str:
    """Create device offer 1FC9 message for any device ID using the exact format from scripts.yaml."""
    device_hex = _device_id_to_hex(device_id)
    
    # Use the exact same payload structure as the working scripts.yaml
    # Format: 0031E0{device_hex}0131E0{device_hex}001298{device_hex}6710E0{device_hex}001FC9{device_hex}
    payload = f"0031E0{device_hex}0131E0{device_hex}001298{device_hex}6710E0{device_hex}001FC9{device_hex}"
    return payload


def _create_device_confirm_1fc9() -> str:
    """Create device confirmation 1FC9 message (always '00')."""
    return "00"


async def _execute_binding_sequence(hass: HomeAssistant, device_id: str, controller_id: str, remote_entity_id: str, zone_number: str):
    """Execute the 1FC9 binding sequence for a CO2 sensor."""
    
    try:
        _LOGGER.info(f"Starting 1FC9 binding sequence for zone {zone_number}: {device_id} → {controller_id}")
        
        # Generate dynamic payloads based on actual device IDs
        device_offer_payload = _create_device_offer_1fc9(device_id)
        device_confirm_payload = _create_device_confirm_1fc9()
        
        _LOGGER.debug(f"Generated payloads - Offer: {device_offer_payload}, Confirm: {device_confirm_payload}")
        
        # Create unique command names for this binding sequence
        offer_cmd = f"bind_offer_{zone_number}"
        confirm_cmd = f"bind_confirm_{zone_number}"
        info_cmd = f"bind_info_{zone_number}"
        request_cmd = f"bind_request_{zone_number}"
        
        # Add all commands first
        commands_to_add = [
            (offer_cmd, f" I --- {device_id} 63:262142 --:------ 1FC9 030 {device_offer_payload}"),
            (confirm_cmd, f" I --- {device_id} {controller_id} --:------ 1FC9 001 {device_confirm_payload}"),
            (info_cmd, f" I --- {device_id} 63:262142 --:------ 10E0 038 000001C8500B0167FEFFFFFFFFFF090307E1564D532D31354331360000000000000000000000"),
            (request_cmd, f"RQ --- {device_id} {controller_id} --:------ 31D9 001 00"),
        ]
        
        for cmd_name, packet_string in commands_to_add:
            try:
                await hass.services.async_call(
                    "ramses_cc", "add_command",
                    {
                        "entity_id": remote_entity_id,
                        "command": cmd_name,
                        "packet_string": packet_string
                    },
                    blocking=True
                )
                _LOGGER.debug(f"Added command '{cmd_name}': {packet_string}")
            except Exception as e:
                _LOGGER.error(f"Failed to add command '{cmd_name}': {e}")
                return
        
        # Give ramses_cc a moment to process all add_command calls
        await hass.async_add_executor_job(time.sleep, 0.5)
        
        # Step 1: Device broadcasts capabilities offer
        await hass.services.async_call(
            "ramses_cc",
            "send_command",
            {
                "command": offer_cmd,
                "num_repeats": 1,
                "delay_secs": 0.5
            },
            target={"entity_id": remote_entity_id}
        )
        
        # Short delay before device confirm (200ms as per scripts.yaml)
        await hass.async_add_executor_job(time.sleep, 0.2)
        
        # Step 2: Device confirms binding
        await hass.services.async_call(
            "ramses_cc",
            "send_command",
            {
                "command": confirm_cmd,
                "num_repeats": 1,
                "delay_secs": 0.5
            },
            target={"entity_id": remote_entity_id}
        )
        
        # Delay before device info broadcast (1 second as per scripts.yaml)
        await hass.async_add_executor_job(time.sleep, 1.0)
        
        # Step 3: Device broadcasts device information
        await hass.services.async_call(
            "ramses_cc",
            "send_command",
            {
                "command": info_cmd,
                "num_repeats": 1,
                "delay_secs": 0.5
            },
            target={"entity_id": remote_entity_id}
        )
        
        # Delay before sensor config request (2 seconds as per scripts.yaml)
        await hass.async_add_executor_job(time.sleep, 2.0)
        
        # Step 4: Device requests sensor configuration
        await hass.services.async_call(
            "ramses_cc",
            "send_command",
            {
                "command": request_cmd,
                "num_repeats": 1,
                "delay_secs": 0.5
            },
            target={"entity_id": remote_entity_id}
        )
        
        _LOGGER.info(f"Completed 1FC9 binding sequence for zone {zone_number}")
        
    except Exception as e:
        _LOGGER.error(f"Error executing binding sequence for zone {zone_number}: {e}", exc_info=True)


async def _cleanup_orphaned_zone_entities(hass: HomeAssistant, entry: ConfigEntry, current_num_zones: int):
    """Remove orphaned zone entities when zone count is reduced."""
    try:
        entity_registry = er.async_get(hass)
        
        # Find all AQI sensors for this integration
        aqi_entities = [
            entity for entity in entity_registry.entities.values()
            if entity.config_entry_id == entry.entry_id
            and entity.domain == "sensor"
            and entity.unique_id and "_aqi_zone_" in entity.unique_id
        ]
        
        # Remove entities for zones that no longer exist
        removed_count = 0
        for entity in aqi_entities:
            # Extract zone number from unique_id (format: "<entry_id>_aqi_zone_<zone_id>")
            try:
                zone_id_str = entity.unique_id.split("_aqi_zone_")[-1]
                zone_id = int(zone_id_str)
                
                # Remove if this zone is beyond the current zone count
                if zone_id > current_num_zones:
                    _LOGGER.info(f"Removing orphaned AQI sensor for zone {zone_id} (unique_id: {entity.unique_id})")
                    entity_registry.async_remove(entity.entity_id)
                    removed_count += 1
                    
            except (ValueError, IndexError) as e:
                _LOGGER.warning(f"Could not parse zone ID from entity unique_id {entity.unique_id}: {e}")
                continue
        
        if removed_count > 0:
            _LOGGER.info(f"Cleaned up {removed_count} orphaned zone entities")
        else:
            _LOGGER.debug("No orphaned zone entities to clean up")
            
    except Exception as e:
        _LOGGER.error(f"Error cleaning up orphaned zone entities: {e}", exc_info=True)


async def _unregister_bind_services(hass: HomeAssistant, entry: ConfigEntry):
    """Unregister bind_co2_n services for this entry."""
    try:
        # Get zone configs to know which services to remove
        cfg = {**entry.data, **entry.options}
        zone_configs = cfg.get(CONF_ZONE_CONFIGS, {})
        
        for zone_id in zone_configs.keys():
            service_name = f"bind_co2_{zone_id}"
            if hass.services.has_service(DOMAIN, service_name):
                hass.services.async_remove(DOMAIN, service_name)
                _LOGGER.info(f"Unregistered service: {DOMAIN}.{service_name}")
                
    except Exception as e:
        _LOGGER.error(f"Error unregistering bind services: {e}")


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry with improved error handling."""
    _LOGGER.debug(f"Starting unload for config entry {entry.entry_id}")
    
    # Remove registered services for this entry
    await _unregister_bind_services(hass, entry)
    
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


def validate_entry_consistency(entry: ConfigEntry) -> bool:
    """
    Validate that the config entry has consistent zone configuration.
    
    Args:
        entry: Config entry to validate
        
    Returns:
        bool: True if consistent, False if validation failed
    """
    try:
        from .config_flow import merge_config_with_validation
        cfg = merge_config_with_validation(entry.data, entry.options, allow_cleanup=False)
        
        zone_configs = cfg.get("zone_configs", {})
        actual_zone_count = len(zone_configs)
        validated_zones = actual_zone_count if zone_configs else int(cfg.get("num_zones", 1))
        
        if actual_zone_count != validated_zones:
            _LOGGER.warning(f"Zone inconsistency detected: validated={validated_zones}, actual={actual_zone_count}")
            return False
            
        # Validate zone IDs are sequential starting from 1
        expected_zones = set(range(1, validated_zones + 1))
        actual_zones = set(zone_configs.keys())
        
        if actual_zones != expected_zones:
            _LOGGER.warning(f"Zone ID mismatch: expected={expected_zones}, actual={actual_zones}")
            return False
            
        _LOGGER.debug(f"Entry validation passed: {validated_zones} zones properly configured")
        return True
        
    except Exception as e:
        _LOGGER.error(f"Entry validation failed: {e}")
        return False


async def async_delayed_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Schedule a delayed reload to prevent hanging during options flow."""
    _LOGGER.info(f"Scheduling delayed reload for config entry {entry.entry_id}")
    
    # Validate entry consistency before reload
    if not validate_entry_consistency(entry):
        _LOGGER.warning("Entry consistency validation failed, but proceeding with reload")
    
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
