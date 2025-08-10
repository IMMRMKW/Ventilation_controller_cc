from datetime import timedelta, datetime
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.const import Platform
from homeassistant.helpers.dispatcher import async_dispatcher_send
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
    CONF_CO2_SENSORS,
    CONF_VOC_SENSORS,
    CONF_PM_SENSORS,
    CONF_SETPOINT,
    CONF_MIN_FAN_OUTPUT,
    CONF_MAX_FAN_OUTPUT,
    CONF_KP,
    CONF_KI_TIMES,
    CONF_UPDATE_INTERVAL,
    CONF_ENTITY_ID,
    CONF_CO2_INDEX,
    CONF_VOC_INDEX,
    CONF_VOC_PPM_INDEX,
    CONF_PM_1_0_INDEX,
    CONF_PM_2_5_INDEX,
    CONF_PM_10_INDEX,
)

_LOGGER = logging.getLogger(__name__)

# Global variable to store last execution time
last_execution_time = None

PLATFORMS = [Platform.SWITCH, Platform.SENSOR]

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
    thresholds_by_dc: dict[str, list[float]],
    previous_values: dict[str, float],
) -> tuple[float, str, float, str]:
    """Return (max_index, worst_entity_id, value, device_class)."""
    worst_idx = 0.0
    worst_entity = ""
    worst_value = 0.0
    worst_dc = ""

    # Build a combined list preserving group to allow fallback mapping
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

    return worst_idx, worst_entity, worst_value, worst_dc

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up PID Ventilation Control from a config entry."""
    # Merge options over data so runtime changes take immediate effect
    cfg = {**entry.data, **entry.options}
    # Get configuration from merged config
    co2_sensors = cfg.get(CONF_CO2_SENSORS, [])
    voc_sensors = cfg.get(CONF_VOC_SENSORS, [])
    pm_sensors = cfg.get(CONF_PM_SENSORS, [])
    setpoint = cfg[CONF_SETPOINT]
    min_fan_output = cfg[CONF_MIN_FAN_OUTPUT]
    max_fan_output = cfg[CONF_MAX_FAN_OUTPUT]
    kp_config = cfg[CONF_KP]
    ki_times = cfg[CONF_KI_TIMES]
    update_interval = cfg[CONF_UPDATE_INTERVAL]
    entity_id = cfg[CONF_ENTITY_ID]
    # Indices (use defaults if not provided)
    co2_index = cfg.get(CONF_CO2_INDEX, DEFAULT_CO2_INDEX)
    voc_index = cfg.get(CONF_VOC_INDEX, DEFAULT_VOC_INDEX)
    voc_ppm_index = cfg.get(CONF_VOC_PPM_INDEX, DEFAULT_VOC_PPM_INDEX)
    pm_1_0_index = cfg.get(CONF_PM_1_0_INDEX, DEFAULT_PM_1_0_INDEX)
    pm_2_5_index = cfg.get(CONF_PM_2_5_INDEX, DEFAULT_PM_2_5_INDEX)
    pm_10_index = cfg.get(CONF_PM_10_INDEX, DEFAULT_PM_10_INDEX)

    thresholds_by_dc = {
        "carbon_dioxide": co2_index,
        "volatile_organic_compounds": voc_index,
        "volatile_organic_compounds_parts": voc_ppm_index,
        "pm1": pm_1_0_index,
        "pm25": pm_2_5_index,
        "pm10": pm_10_index,
    }
    
    _LOGGER.info(
        f"Ventilation setup - CO2: {len(co2_sensors)}, VOC: {len(voc_sensors)}, PM: {len(pm_sensors)}, Setpoint: {setpoint}"
    )
    _LOGGER.info(f"Fan range: {min_fan_output}-{max_fan_output}, Kp: {kp_config}, Update interval: {update_interval}s")
  
    # Fan discrete speeds [percent]
    fan_speeds = np.arange(min_fan_output, max_fan_output + 1e-6, 1)

    # PID coefficients
    fan_diff = fan_speeds.max() - fan_speeds.min()
    Kp = kp_config
    Ki = [fan_diff / ki_time for ki_time in ki_times]
    
    _LOGGER.info(f"Calculated Ki values: {Ki}")
    #Kd = 0.005

    integral = 0.0
    prev_error = 0.0

    # Previous values cache per entity
    previous_values: dict[str, float] = {}

    async def pid_control(now):
        nonlocal integral, prev_error, previous_values
        global last_execution_time
        # Skip if controller disabled via switch
        domain_data = hass.data.get(DOMAIN, {})
        entry_data = domain_data.get(entry.entry_id, {}) if isinstance(domain_data, dict) else {}
        if not entry_data.get("enabled", True):
            # Reset time baseline to avoid integral spike when re-enabled
            last_execution_time = datetime.now()
            return
         
        # Calculate time difference since last execution
        current_time = datetime.now()
        if last_execution_time is not None:
            time_diff = (current_time - last_execution_time).total_seconds()
            _LOGGER.info(f"Time since last execution: {time_diff:.2f} seconds")
        else:
            _LOGGER.info("First execution - no previous time to compare")
            time_diff = float(update_interval)  # Use configured interval for first run
        
        # Update last execution time
        last_execution_time = current_time

        # Compute worst AQ index across all sensors
        aqi, worst_entity, worst_value, worst_dc = _worst_air_quality_index(
            hass,
            co2_sensors,
            voc_sensors,
            pm_sensors,
            thresholds_by_dc,
            previous_values,
        )
        air_quality_index = aqi

        # Update shared state and notify subscribers (sensor)
        try:
            ed = hass.data.get(DOMAIN, {}).get(entry.entry_id)
            if isinstance(ed, dict):
                ed["aqi"] = air_quality_index
                ed["worst_entity"] = worst_entity
                ed["worst_value"] = worst_value
                ed["worst_dc"] = worst_dc
            async_dispatcher_send(
                hass,
                f"{SIGNAL_AQI_UPDATED}_{entry.entry_id}",
                {
                    "aqi": air_quality_index,
                    "worst_entity": worst_entity,
                    "worst_value": worst_value,
                    "worst_dc": worst_dc,
                },
            )
        except Exception:  # best-effort notification
            pass

        idx = min(int(round(air_quality_index)), len(Ki) - 1)
        Ki_index_based = Ki[idx]
        error       = air_quality_index - setpoint
        integral   += error * Ki_index_based * time_diff
        prev_error  = error

        # PID output (desired removal rate)
        pid_output = Kp * error + integral + fan_speeds.min() # + Kd * derivative

        # Map to discrete fan speed and handle integral windup
        if pid_output < fan_speeds[0]:
            rate = int(fan_speeds[0])
            integral -= error * Ki_index_based * time_diff  # Remove the integral contribution that caused windup
        elif pid_output > fan_speeds[-1]:
            rate = int(fan_speeds[-1])
            integral -= error * Ki_index_based * time_diff  # Remove the integral contribution that caused windup
        else:
            # Round to nearest integer within allowed range
            rate = int(round(float(np.clip(pid_output, fan_speeds[0], fan_speeds[-1]))))
        #command = percent_to_line(rate)
        command = rate
        _LOGGER.info(f"Sending command: {command}")

        await hass.services.async_call(
            "ramses_cc", "send_command",
            {
                "num_repeats": 3,
                "delay_secs": 0.05,
                "entity_id": entity_id,
                "command": command
            },
            blocking=True
        )

        # Compute and publish rate percentage
        try:
            span = max_fan_output - min_fan_output
            pct = 0.0 if span <= 0 else round((rate - min_fan_output) * 100.0 / span, 2)
            ed = hass.data.get(DOMAIN, {}).get(entry.entry_id)
            if isinstance(ed, dict):
                ed["rate_pct"] = pct
            async_dispatcher_send(
                hass,
                f"{SIGNAL_RATE_UPDATED}_{entry.entry_id}",
                {"rate_pct": pct, "rate": rate},
            )
        except Exception:
            pass

        _LOGGER.info(
            f"PID: Worst={worst_entity} (dc={worst_dc}, val={worst_value}), AQI={air_quality_index:.2f}, Error={error:.2f}, Fan={rate}%, Integral={integral:.2f}"
        )

    # Store the remove callback and runtime state in hass.data for cleanup and control
    remove_listener = async_track_time_interval(hass, pid_control, timedelta(seconds=update_interval))
    
    # Add options update listener
    entry.add_update_listener(async_reload_entry)
    
    if DOMAIN not in hass.data:
        hass.data[DOMAIN] = {}
    hass.data[DOMAIN][entry.entry_id] = {
        "enabled": True,
        "remove_listener": remove_listener,
        "aqi": 0.0,
        "worst_entity": None,
        "worst_value": None,
        "worst_dc": None,
        "rate_pct": 0.0,
    }

    # Ensure switch platform is set up after state exists
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
 
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    # Unload platforms
    await hass.config_entries.async_unload_platforms(entry, PLATFORMS)

    # Clean up timer
    data = hass.data[DOMAIN].pop(entry.entry_id, None)
    if isinstance(data, dict):
        remove_listener = data.get("remove_listener")
        if remove_listener:
            remove_listener()

    if not hass.data[DOMAIN]:
        hass.data.pop(DOMAIN)
    
    return True


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload config entry."""
    await async_unload_entry(hass, entry)
    await async_setup_entry(hass, entry)


async def async_setup(hass, config):
    """Set up the ventilation component from YAML (deprecated)."""
    # This is kept for backward compatibility but should not be used
    # All new installations should use the config flow
    return True
