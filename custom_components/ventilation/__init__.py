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
    CONF_CO2_SENSORS,
    CONF_VOC_SENSORS,
    CONF_PM_SENSORS,
    CONF_SETPOINT,
    CONF_MIN_FAN_OUTPUT,
    CONF_MAX_FAN_OUTPUT,
    CONF_KP,
    CONF_KI_TIMES,
    CONF_UPDATE_INTERVAL,
    CONF_REMOTE_DEVICE,
    CONF_FAN_DEVICE,
    CONF_CO2_INDEX,
    CONF_VOC_INDEX,
    CONF_VOC_PPM_INDEX,
    CONF_PM_1_0_INDEX,
    CONF_PM_2_5_INDEX,
    CONF_PM_10_INDEX,
)

_LOGGER = logging.getLogger(__name__)

PLATFORMS = [Platform.SWITCH, Platform.SENSOR, Platform.NUMBER]

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

def format_fan_command(remote_serial: str, fan_serial: str, rate: int) -> str:
    """
    Convert rate to ramses_cc command format.
    
    Args:
        remote_serial: Serial number of the remote device (e.g., "29:162275")
        fan_serial: Serial number of the fan device (e.g., "32:146231") 
        rate: Fan power rate (0-255)
        
    Returns:
        Formatted command string like " I --- 29:162275 32:146231 --:------ 31E0 008 00000A000100AA00"
    """
    # Convert rate to hex (2 digits, uppercase)
    rate_hex = f"{rate:02X}"
    
    # Build the command string
    command = f" I --- {remote_serial} {fan_serial} --:------ 31E0 008 0000{rate_hex}000100AA00"
    
    return command

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
        # Skip if controller disabled via switch
        domain_data = hass.data.get(DOMAIN, {})
        entry_data = domain_data.get(entry.entry_id, {}) if isinstance(domain_data, dict) else {}
        if not entry_data.get("enabled", True):
            # Reset time baseline to avoid integral spike when re-enabled
            entry_data["last_execution_time"] = datetime.now()
            return
        
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
        error       = air_quality_index - current_setpoint
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
        
        # Format the command using the extracted serial numbers
        if remote_serial_number and fan_serial_number:
            command_name = str(rate)
            
            # Get the learned commands set for this entry
            domain_data = hass.data.get(DOMAIN, {})
            entry_data = domain_data.get(entry.entry_id, {})
            learned_commands = entry_data.get("learned_commands", set())
            
            # Check if this rate command has been learned before
            if command_name not in learned_commands:
                # Need to learn the command first
                formatted_command = format_fan_command(remote_serial_number, fan_serial_number, rate)
                _LOGGER.info(f"Learning new command '{command_name}': {formatted_command}")
                
                try:
                    # await hass.services.async_call(
                    #     "ramses_cc", "learn_command",
                    #     {
                    #         "entity_id": remote_entity_id,
                    #         "command": command_name,
                    #         "packet": formatted_command
                    #     },
                    #     blocking=True
                    # )
                    
                    # Mark this command as learned
                    learned_commands.add(command_name)
                    entry_data["learned_commands"] = learned_commands
                    _LOGGER.info(f"Successfully learned command '{command_name}'")
                    
                except Exception as e:
                    _LOGGER.error(f"Failed to learn command '{command_name}': {e}")
                    return
            
            # Send the command using the learned command name
            _LOGGER.info(f"Sending command '{command_name}' (rate: {rate})")
            
            try:
                await hass.services.async_call(
                    "ramses_cc", "send_command",
                    {
                        "entity_id": remote_entity_id,
                        "command": command_name
                    },
                    blocking=True
                )
            except Exception as e:
                _LOGGER.error(f"Failed to send command '{command_name}': {e}")
                return
                
        else:
            # Fallback to old method if serial numbers not available
            _LOGGER.warning("Serial numbers not available, using fallback command method")
            await hass.services.async_call(
                "ramses_cc", "send_command",
                {
                    "num_repeats": 3,
                    "delay_secs": 0.05,
                    "entity_id": remote_entity_id,
                    "command": rate
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
            f"PID: Worst={worst_entity} (dc={worst_dc}, val={worst_value}), AQI={air_quality_index:.2f}, Setpoint={current_setpoint:.2f}, Error={error:.2f}, Fan={rate}%, Integral={integral:.2f}"
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
        "worst_entity": None,
        "worst_value": None,
        "worst_dc": None,
        "rate_pct": 0.0,
        "remote_device_id": remote_device_id,
        "remote_entity_id": remote_entity_id,
        "remote_serial_number": remote_serial_number,
        "fan_device_id": fan_device_id,
        "fan_serial_number": fan_serial_number,
        "learned_commands": set(),  # Track which rate commands have been learned
        "last_execution_time": None,  # Instance-specific last execution time
        "current_setpoint": setpoint,  # Initialize with config setpoint
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
