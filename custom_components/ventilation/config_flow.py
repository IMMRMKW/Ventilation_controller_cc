"""Config flow for PID Ventilation Control integration."""
from __future__ import annotations

import voluptuous as vol
from typing import Any
import logging

from homeassistant import config_entries
from homeassistant.core import HomeAssistant, callback
from homeassistant.data_entry_flow import FlowResult
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import selector
from homeassistant.helpers.selector import BooleanSelector
import homeassistant.helpers.config_validation as cv
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import device_registry as dr

from .const import (
    DOMAIN,
    DEFAULT_CO2_INDEX,
    DEFAULT_VOC_PPM_INDEX,
    DEFAULT_VOC_INDEX,
    DEFAULT_PM_1_0_INDEX,
    DEFAULT_PM_2_5_INDEX,
    DEFAULT_PM_10_INDEX,
    DEFAULT_HUMIDITY_INDEX,
    CONF_ZONE_SENSOR_ID,
)
_LOGGER = logging.getLogger(__name__)

def get_zone_options(num_zones: int) -> list[selector.SelectOptionDict]:
    """Generate zone selection options based on number of zones."""
    options = []
    for i in range(1, num_zones + 1):
        options.append(selector.SelectOptionDict(value=str(i), label=f"Zone {i}"))
    return options


def get_entity_display_name(hass: HomeAssistant, entity_id: str) -> str:
    """Get friendly display name for an entity."""
    if hass is None:
        return entity_id.split('.')[-1].replace('_', ' ').title()
    
    registry = er.async_get(hass)
    entity_entry = registry.async_get(entity_id)
    
    if entity_entry and entity_entry.name:
        return entity_entry.name
    
    # Fallback to state object
    state = hass.states.get(entity_id)
    if state and state.attributes.get("friendly_name"):
        return state.attributes["friendly_name"]
    
    # Final fallback to entity_id formatted nicely
    return entity_id.split('.')[-1].replace('_', ' ').title()


def merge_config_with_validation(entry_data: dict, entry_options: dict, allow_cleanup: bool = False) -> dict:
    """
    Merge entry data and options with proper zone validation.
    
    This replaces the simple {**entry_data, **entry_options} pattern to prevent
    sensor duplication bugs by ensuring zone configs stay consistent.
    
    Args:
        entry_data: Configuration entry data
        entry_options: Configuration entry options  
        allow_cleanup: If True, allows cleanup of excess zone configs (only use in options flow)
    """
    # Merge configs with options taking precedence (standard behavior)
    merged_config = {**entry_data, **entry_options}
    
    # Get zone configurations and determine authoritative zone count
    zone_configs = merged_config.get("zone_configs", {})
    explicit_num_zones = merged_config.get("num_zones")
    
    # Determine the authoritative number of zones:
    # Priority: options num_zones > zone_configs count > data num_zones > fallback
    if "num_zones" in entry_options:
        target_num_zones = int(entry_options["num_zones"])
        _LOGGER.debug(f"Using num_zones from options: {target_num_zones}")
    elif zone_configs:
        target_num_zones = len(zone_configs) 
        _LOGGER.debug(f"Using zone_configs count: {target_num_zones}")
    elif explicit_num_zones:
        target_num_zones = int(explicit_num_zones)
        _LOGGER.debug(f"Using explicit num_zones: {target_num_zones}")
    else:
        target_num_zones = 1
        _LOGGER.debug("Using fallback num_zones: 1")
    
    # Only clean up zone configs if explicitly allowed (to prevent accidental data loss)
    if allow_cleanup and zone_configs and target_num_zones < len(zone_configs):
        # Extra safety: Verify that we have valid device configurations before removing zones
        zones_to_remove = set(zone_configs.keys()) - set(range(1, target_num_zones + 1))
        safe_to_remove = True
        
        for zone_id in zones_to_remove:
            zone_config = zone_configs.get(zone_id, {})
            # Check if zone has important device configurations
            if zone_config.get("id_to") or zone_config.get("device_to"):
                _LOGGER.warning(f"Zone {zone_id} has device config (id_to: {zone_config.get('id_to')}) - requiring explicit confirmation")
                safe_to_remove = False
        
        if safe_to_remove:
            zones_to_keep = set(range(1, target_num_zones + 1))
            cleaned_zone_configs = {
                zone_id: config for zone_id, config in zone_configs.items()
                if isinstance(zone_id, int) and zone_id in zones_to_keep
            }
            
            # Log cleanup for debugging
            removed_zones = set(zone_configs.keys()) - set(cleaned_zone_configs.keys())
            if removed_zones:
                _LOGGER.warning(f"Cleaned up zone configs (user requested): {sorted(removed_zones)}")
            
            merged_config["zone_configs"] = cleaned_zone_configs
        else:
            _LOGGER.error(f"Cannot remove zones with device configurations - preserving all zones")
            target_num_zones = len(zone_configs)
    elif zone_configs and target_num_zones < len(zone_configs):
        # Log potential inconsistency but don't auto-cleanup
        _LOGGER.warning(f"Zone config inconsistency detected: {len(zone_configs)} configs vs {target_num_zones} zones - preserving existing configs to prevent data loss")
        # Use the actual zone config count to preserve user data
        target_num_zones = len(zone_configs)
    
    # Ensure num_zones reflects the validated count
    merged_config["num_zones"] = target_num_zones
    
    return merged_config

async def get_device_serial_number(hass: HomeAssistant, device_id: str | None) -> str | None:
    """Extract serial number from a device ID."""
    if not device_id:
        return None
    try:
        device_registry = dr.async_get(hass)
        device = device_registry.async_get(device_id)
        if device:
            # Extract serial number from device identifiers
            for domain, identifier in device.identifiers:
                if isinstance(identifier, str) and len(identifier) > 6:
                    return identifier
    except Exception as e:
        _LOGGER.warning(f"Failed to extract serial number from device {device_id}: {e}")
    return None

def get_device_selection_schema():
    """Generate schema for device selection (remote device and number of zones)."""
    return vol.Schema({
        vol.Required("remote_device"): selector.DeviceSelector(
            selector.DeviceSelectorConfig(integration="ramses_cc")
        ),
        vol.Optional("num_zones", default="1"): selector.SelectSelector(
            selector.SelectSelectorConfig(
                options=[
                    selector.SelectOptionDict(value="1", label="1 Zone"),
                    selector.SelectOptionDict(value="2", label="2 Zones"), 
                    selector.SelectOptionDict(value="3", label="3 Zones"),
                    selector.SelectOptionDict(value="4", label="4 Zones"),
                    selector.SelectOptionDict(value="5", label="5 Zones"),
                ],
                mode=selector.SelectSelectorMode.DROPDOWN
            )
        ),
    }, extra=vol.ALLOW_EXTRA)


def get_fan_settings_schema():
    """Generate schema for fan settings only."""
    return vol.Schema({
        vol.Required("min_fan_output", default=0): vol.Coerce(int),
        vol.Required("max_fan_output", default=255): vol.Coerce(int),
        vol.Optional("back", default=False): selector.BooleanSelector(),
    }, extra=vol.ALLOW_EXTRA)


def get_basic_schema():
    """Generate schema for fan settings."""
    return vol.Schema({
        vol.Required("min_fan_output", default=0): vol.Coerce(int),
        vol.Required("max_fan_output", default=255): vol.Coerce(int),
        vol.Required("remote_device"): selector.DeviceSelector(
            selector.DeviceSelectorConfig(integration="ramses_cc")
        ),
    }, extra=vol.ALLOW_EXTRA)


def get_zone_config_schema(zone_number: int, current_config: dict = None, remote_device_id: str = None):
    """Generate schema for zone configuration (device selection and fan settings)."""
    if current_config is None:
        current_config = {}
    
    # Use device IDs if available, otherwise fall back to remote device for id_from
    default_device_from = current_config.get("device_from", remote_device_id)
    default_device_to = current_config.get("device_to", None)
    
    # Calculate default sensor ID: 255 for zone 1, 254 for zone 2, etc.
    default_sensor_id = current_config.get("sensor_id", 256 - zone_number)
    
    # Convert stored 0-255 values to percentages for display (if they exist)
    current_min_pct = current_config.get("min_fan_rate", 0)
    current_max_pct = current_config.get("max_fan_rate", 100)
    
    # If the stored values are in 0-255 range, convert to percentage
    if current_min_pct > 100:
        current_min_pct = round((current_min_pct / 255) * 100)
    if current_max_pct > 100:
        current_max_pct = round((current_max_pct / 255) * 100)
    
    try:
        # Build schema fields conditionally to avoid None defaults
        schema_fields = {}
        
        # Add device_from field
        if default_device_from:
            schema_fields[vol.Optional("device_from", default=default_device_from)] = selector.DeviceSelector(
                selector.DeviceSelectorConfig(integration="ramses_cc")
            )
        else:
            schema_fields[vol.Optional("device_from")] = selector.DeviceSelector(
                selector.DeviceSelectorConfig(integration="ramses_cc")
            )
        
        # Add device_to field 
        if default_device_to:
            schema_fields[vol.Optional("device_to", default=default_device_to)] = selector.DeviceSelector(
                selector.DeviceSelectorConfig(integration="ramses_cc")
            )
        else:
            schema_fields[vol.Optional("device_to")] = selector.DeviceSelector(
                selector.DeviceSelectorConfig(integration="ramses_cc")
            )
        
        # Add other fields (now using percentages)
        schema_fields[vol.Required("sensor_id", default=default_sensor_id)] = vol.Coerce(int)
        schema_fields[vol.Required("min_fan_rate", default=current_min_pct)] = vol.Coerce(int)  
        schema_fields[vol.Required("max_fan_rate", default=current_max_pct)] = vol.Coerce(int)
        schema_fields[vol.Optional("back", default=False)] = selector.BooleanSelector()
        
        return vol.Schema(schema_fields, extra=vol.ALLOW_EXTRA)
    except Exception as e:
        _LOGGER.error(f"Error creating zone config schema: {e}")
        # Fallback schema without device selectors 
        return vol.Schema({
            vol.Optional("device_from", default=""): cv.string,
            vol.Optional("device_to", default=""): cv.string,
            vol.Required("sensor_id", default=default_sensor_id): vol.Coerce(int),
            vol.Required("min_fan_rate", default=current_min_pct): vol.Coerce(int),
            vol.Required("max_fan_rate", default=current_max_pct): vol.Coerce(int),
            vol.Optional("back", default=False): selector.BooleanSelector(),
        }, extra=vol.ALLOW_EXTRA)


def get_pid_parameters_schema():
    """Generate schema for PID parameters (including Ki times)."""
    # Custom validator for locale-aware float input with range validation
    def validate_float_with_locale(min_val: float, max_val: float):
        return vol.All(
            vol.Coerce(str),
            lambda v: str(v).replace(',', '.'),  # Convert comma to dot for locale compatibility
            vol.Coerce(float),
            vol.Range(min=min_val, max=max_val),
        )
    
    return vol.Schema({
        vol.Required("setpoint", default=1.2): validate_float_with_locale(0.0, 5.0),
        vol.Required("kp", default=25.5): validate_float_with_locale(0.1, 1000.0),
        vol.Required("ki_times", default="3600,1800,900,450,225,150"): cv.string,
        vol.Required("update_interval", default=300): vol.Coerce(int),
        vol.Optional("back", default=False): selector.BooleanSelector(),
    }, extra=vol.ALLOW_EXTRA)


def get_sensor_types_schema():
    """Generate schema for selecting sensor types."""
    return vol.Schema({
        vol.Optional("use_co2_sensors", default=True): cv.boolean,
        vol.Optional("use_voc_sensors", default=False): cv.boolean,
        vol.Optional("use_pm_sensors", default=False): cv.boolean,
        vol.Optional("use_humidity_sensors", default=False): cv.boolean,
        vol.Optional("back", default=False): selector.BooleanSelector(),
    }, extra=vol.ALLOW_EXTRA)


def get_co2_sensors_selection_schema(current_sensors=None):
    """Generate schema for CO2 sensor selection only."""
    default_sensors = current_sensors if current_sensors else []
    
    return vol.Schema({
        vol.Required("co2_sensors", default=default_sensors): selector.EntitySelector(
            selector.EntitySelectorConfig(
                domain="sensor",
                device_class="carbon_dioxide",
                multiple=True,
            )
        ),
        vol.Optional("back", default=False): selector.BooleanSelector(),
    }, extra=vol.ALLOW_EXTRA)

def get_voc_sensors_selection_schema(current_sensors=None):
    """Generate schema for VOC sensor selection only."""
    default_sensors = current_sensors if current_sensors else []
    
    return vol.Schema({
        vol.Required("voc_sensors", default=default_sensors): selector.EntitySelector(
            selector.EntitySelectorConfig(
                domain="sensor",
                device_class=["volatile_organic_compounds", "volatile_organic_compounds_parts"],
                multiple=True,
            )
        ),
        vol.Optional("back", default=False): selector.BooleanSelector(),
    }, extra=vol.ALLOW_EXTRA)

def get_pm_sensors_selection_schema(current_sensors=None):
    """Generate schema for PM sensor selection only."""
    default_sensors = current_sensors if current_sensors else []
    
    return vol.Schema({
        vol.Required("pm_sensors", default=default_sensors): selector.EntitySelector(
            selector.EntitySelectorConfig(
                domain="sensor",
                device_class=["pm1", "pm25", "pm10"],
                multiple=True,
            )
        ),
        vol.Optional("back", default=False): selector.BooleanSelector(),
    }, extra=vol.ALLOW_EXTRA)

def get_humidity_sensors_selection_schema(current_sensors=None):
    """Generate schema for humidity sensor selection only."""
    default_sensors = current_sensors if current_sensors else []
    
    return vol.Schema({
        vol.Required("humidity_sensors", default=default_sensors): selector.EntitySelector(
            selector.EntitySelectorConfig(
                domain="sensor",
                device_class="humidity",
                multiple=True,
            )
        ),
        vol.Optional("back", default=False): selector.BooleanSelector(),
    }, extra=vol.ALLOW_EXTRA)

def get_co2_sensors_zones_schema(hass: HomeAssistant, sensors: list, current_zones=None, num_zones=1):
    """Generate schema for CO2 sensor zone assignments with sensor names."""
    if not sensors or num_zones <= 1:
        return None
    
    default_zones = current_zones if current_zones else ["1"] * len(sensors)
    zone_options = get_zone_options(num_zones)
    
    schema_dict = {}
    
    for i, sensor in enumerate(sensors):
        current_zone = str(default_zones[i]) if i < len(default_zones) else "1"
        field_key = f"co2_sensor_{i}_zone"
        
        schema_dict[vol.Required(field_key, default=current_zone)] = selector.SelectSelector(
            selector.SelectSelectorConfig(
                options=zone_options,
                mode=selector.SelectSelectorMode.DROPDOWN,
            )
        )
    
    schema_dict[vol.Optional("back", default=False)] = selector.BooleanSelector()
    
    return vol.Schema(schema_dict, extra=vol.ALLOW_EXTRA)

def get_voc_sensors_zones_schema(hass: HomeAssistant, sensors: list, current_zones=None, num_zones=1):
    """Generate schema for VOC sensor zone assignments with sensor names."""
    if not sensors or num_zones <= 1:
        return None
    
    default_zones = current_zones if current_zones else ["1"] * len(sensors)
    zone_options = get_zone_options(num_zones)
    
    schema_dict = {}
    
    for i, sensor in enumerate(sensors):
        current_zone = str(default_zones[i]) if i < len(default_zones) else "1"
        field_key = f"voc_sensor_{i}_zone"
        
        schema_dict[vol.Required(field_key, default=current_zone)] = selector.SelectSelector(
            selector.SelectSelectorConfig(
                options=zone_options,
                mode=selector.SelectSelectorMode.DROPDOWN,
            )
        )
    
    schema_dict[vol.Optional("back", default=False)] = selector.BooleanSelector()
    
    return vol.Schema(schema_dict, extra=vol.ALLOW_EXTRA)

def get_pm_sensors_zones_schema(hass: HomeAssistant, sensors: list, current_zones=None, num_zones=1):
    """Generate schema for PM sensor zone assignments with sensor names."""
    if not sensors or num_zones <= 1:
        return None
    
    default_zones = current_zones if current_zones else ["1"] * len(sensors)
    zone_options = get_zone_options(num_zones)
    
    schema_dict = {}
    
    for i, sensor in enumerate(sensors):
        current_zone = str(default_zones[i]) if i < len(default_zones) else "1"
        field_key = f"pm_sensor_{i}_zone"
        
        schema_dict[vol.Required(field_key, default=current_zone)] = selector.SelectSelector(
            selector.SelectSelectorConfig(
                options=zone_options,
                mode=selector.SelectSelectorMode.DROPDOWN,
            )
        )
    
    schema_dict[vol.Optional("back", default=False)] = selector.BooleanSelector()
    
    return vol.Schema(schema_dict, extra=vol.ALLOW_EXTRA)

def get_humidity_sensors_zones_schema(hass: HomeAssistant, sensors: list, current_zones=None, num_zones=1):
    """Generate schema for humidity sensor zone assignments with sensor names."""
    if not sensors or num_zones <= 1:
        return None
    
    default_zones = current_zones if current_zones else ["1"] * len(sensors)
    zone_options = get_zone_options(num_zones)
    
    schema_dict = {}
    
    for i, sensor in enumerate(sensors):
        current_zone = str(default_zones[i]) if i < len(default_zones) else "1"
        field_key = f"humidity_sensor_{i}_zone"
        
        schema_dict[vol.Required(field_key, default=current_zone)] = selector.SelectSelector(
            selector.SelectSelectorConfig(
                options=zone_options,
                mode=selector.SelectSelectorMode.DROPDOWN,
            )
        )
    
    schema_dict[vol.Optional("back", default=False)] = selector.BooleanSelector()
    
    return vol.Schema(schema_dict, extra=vol.ALLOW_EXTRA)


def get_co2_sensors_schema(hass: HomeAssistant = None, current_sensors=None):
    """Generate schema for CO2 sensors with multiple entity selector."""
    # Get current sensors as default
    default_sensors = current_sensors if current_sensors else []
    
    schema_dict = {
        vol.Required("co2_sensors", default=default_sensors): selector.EntitySelector(
            selector.EntitySelectorConfig(
                domain="sensor",
                device_class="carbon_dioxide",
                multiple=True,
            )
        ),
        vol.Optional("back", default=False): selector.BooleanSelector(),
    }
    
    return vol.Schema(schema_dict, extra=vol.ALLOW_EXTRA)


def get_voc_sensors_with_zones_schema(hass: HomeAssistant = None, current_sensors=None, current_zones=None, num_zones=1):
    """Generate schema for VOC sensors with individual zone selectors."""
    # Get current sensors and zones as defaults
    default_sensors = current_sensors if current_sensors else []
    default_zones = current_zones if current_zones else []
    
    # Create sensor selector
    schema_dict = {
        vol.Required("voc_sensors", default=default_sensors): selector.EntitySelector(
            selector.EntitySelectorConfig(
                domain="sensor",
                device_class=["volatile_organic_compounds", "volatile_organic_compounds_parts"],
                multiple=True,
            )
        ),
    }
    
    # Add individual zone selectors for each currently selected sensor if multiple zones
    if num_zones > 1 and default_sensors:
        zone_options = get_zone_options(num_zones)
        
        for i, sensor in enumerate(default_sensors):
            # Get sensor friendly name for the selector label
            sensor_name = get_entity_display_name(hass, sensor)
            current_zone = str(default_zones[i]) if i < len(default_zones) else "1"
            
            schema_dict[vol.Required(f"voc_sensor_{i}_zone", default=current_zone)] = selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=zone_options,
                    mode=selector.SelectSelectorMode.DROPDOWN,
                )
            )
    
    schema_dict[vol.Optional("back", default=False)] = selector.BooleanSelector()
    
    return vol.Schema(schema_dict, extra=vol.ALLOW_EXTRA)


def get_voc_sensors_schema(hass: HomeAssistant = None, current_sensors=None):
    """Generate schema for VOC sensors with multiple entity selector."""
    # Get current sensors as default
    default_sensors = current_sensors if current_sensors else []
    
    schema_dict = {
        vol.Required("voc_sensors", default=default_sensors): selector.EntitySelector(
            selector.EntitySelectorConfig(
                domain="sensor",
                device_class=["volatile_organic_compounds", "volatile_organic_compounds_parts"],
                multiple=True,
            )
        ),
        vol.Optional("back", default=False): selector.BooleanSelector(),
    }
    
    return vol.Schema(schema_dict, extra=vol.ALLOW_EXTRA)


def get_pm_sensors_with_zones_schema(hass: HomeAssistant = None, current_sensors=None, current_zones=None, num_zones=1):
    """Generate schema for PM sensors with individual zone selectors."""
    # Get current sensors and zones as defaults
    default_sensors = current_sensors if current_sensors else []
    default_zones = current_zones if current_zones else []
    
    # Create sensor selector
    schema_dict = {
        vol.Required("pm_sensors", default=default_sensors): selector.EntitySelector(
            selector.EntitySelectorConfig(
                domain="sensor",
                device_class=["pm1", "pm10", "pm25"],
                multiple=True,
            )
        ),
    }
    
    # Add individual zone selectors for each currently selected sensor if multiple zones
    if num_zones > 1 and default_sensors:
        zone_options = get_zone_options(num_zones)
        
        for i, sensor in enumerate(default_sensors):
            # Get sensor friendly name for the selector label
            sensor_name = get_entity_display_name(hass, sensor)
            current_zone = str(default_zones[i]) if i < len(default_zones) else "1"
            
            schema_dict[vol.Required(f"pm_sensor_{i}_zone", default=current_zone)] = selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=zone_options,
                    mode=selector.SelectSelectorMode.DROPDOWN,
                )
            )
    
    schema_dict[vol.Optional("back", default=False)] = selector.BooleanSelector()
    
    return vol.Schema(schema_dict, extra=vol.ALLOW_EXTRA)


def get_pm_sensors_schema(hass: HomeAssistant = None, current_sensors=None):
    """Generate schema for Particulate Matter sensors with multiple entity selector."""
    # Get current sensors as default
    default_sensors = current_sensors if current_sensors else []
    
    schema_dict = {
        vol.Required("pm_sensors", default=default_sensors): selector.EntitySelector(
            selector.EntitySelectorConfig(
                domain="sensor",
                device_class=["pm1", "pm10", "pm25"],
                multiple=True,
            )
        ),
        vol.Optional("back", default=False): selector.BooleanSelector(),
    }
    
    return vol.Schema(schema_dict, extra=vol.ALLOW_EXTRA)


def get_humidity_sensors_with_zones_schema(hass: HomeAssistant = None, current_sensors=None, current_zones=None, num_zones=1):
    """Generate schema for humidity sensors with individual zone selectors."""
    # Get current sensors and zones as defaults
    default_sensors = current_sensors if current_sensors else []
    default_zones = current_zones if current_zones else []
    
    # Create sensor selector
    schema_dict = {
        vol.Required("humidity_sensors", default=default_sensors): selector.EntitySelector(
            selector.EntitySelectorConfig(
                domain="sensor",
                device_class="humidity",
                multiple=True,
            )
        ),
    }
    
    # Add individual zone selectors for each currently selected sensor if multiple zones
    if num_zones > 1 and default_sensors:
        zone_options = get_zone_options(num_zones)
        
        for i, sensor in enumerate(default_sensors):
            # Get sensor friendly name for the selector label
            sensor_name = get_entity_display_name(hass, sensor)
            current_zone = str(default_zones[i]) if i < len(default_zones) else "1"
            
            schema_dict[vol.Required(f"humidity_sensor_{i}_zone", default=current_zone)] = selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=zone_options,
                    mode=selector.SelectSelectorMode.DROPDOWN,
                )
            )
    
    schema_dict[vol.Optional("back", default=False)] = selector.BooleanSelector()
    
    return vol.Schema(schema_dict, extra=vol.ALLOW_EXTRA)


def get_humidity_sensors_schema(hass: HomeAssistant = None, current_sensors=None):
    """Generate schema for Humidity sensors with multiple entity selector."""
    # Get current sensors as default
    default_sensors = current_sensors if current_sensors else []
    
    schema_dict = {
        vol.Required("humidity_sensors", default=default_sensors): selector.EntitySelector(
            selector.EntitySelectorConfig(
                domain="sensor",
                device_class="humidity",
                multiple=True,
            )
        ),
        vol.Optional("back", default=False): selector.BooleanSelector(),
    }
    
    return vol.Schema(schema_dict, extra=vol.ALLOW_EXTRA)


def get_air_quality_indices_schema(sensor_types):
    """Generate schema for air quality indices based on selected sensor types."""
    schema_dict = {}
    
    # Add CO2 index if CO2 sensors are enabled
    if sensor_types.get("use_co2_sensors"):
        schema_dict[vol.Required("co2_index", default=",".join(map(str, DEFAULT_CO2_INDEX)))] = cv.string
    
    # Add VOC index if VOC sensors are enabled
    if sensor_types.get("use_voc_sensors"):
        schema_dict[vol.Required("voc_index", default=",".join(map(str, DEFAULT_VOC_INDEX)))] = cv.string
    
    # Add VOC PPM index if VOC PPM sensors are enabled
    if sensor_types.get("use_voc_sensors"):
        schema_dict[vol.Required("voc_ppm_index", default=",".join(map(str, DEFAULT_VOC_PPM_INDEX)))] = cv.string
    
    # Add PM indices if PM sensors are enabled
    if sensor_types.get("use_pm_sensors"):
        schema_dict[vol.Required("pm_1_0_index", default=",".join(map(str, DEFAULT_PM_1_0_INDEX)))] = cv.string
        schema_dict[vol.Required("pm_2_5_index", default=",".join(map(str, DEFAULT_PM_2_5_INDEX)))] = cv.string
        schema_dict[vol.Required("pm_10_index", default=",".join(map(str, DEFAULT_PM_10_INDEX)))] = cv.string
    
    # Add Humidity index if Humidity sensors are enabled
    if sensor_types.get("use_humidity_sensors"):
        schema_dict[vol.Required("humidity_index", default=",".join(map(str, DEFAULT_HUMIDITY_INDEX)))] = cv.string
    
    # Add back button
    schema_dict[vol.Optional("back", default=False)] = selector.BooleanSelector()
    
    return vol.Schema(schema_dict, extra=vol.ALLOW_EXTRA)


async def validate_device_selection_input(hass: HomeAssistant, data: dict[str, Any]) -> dict[str, Any]:
    """Validate the device selection input."""
    
    # Validate remote_device is provided
    if not data.get("remote_device"):
        raise InvalidEntity("Remote device must be selected")
    
    # Validate num_zones - use default if not provided
    num_zones = data.get("num_zones", 1)
    try:
        zones_int = int(num_zones)
        if not (1 <= zones_int <= 5):
            raise InvalidEntity("Number of zones must be between 1 and 5")
    except (ValueError, TypeError):
        raise InvalidEntity("Number of zones must be a valid number")
    
    return {"title": "PID Ventilation Control"}


async def validate_fan_settings_input(hass: HomeAssistant, data: dict[str, Any]) -> dict[str, Any]:
    """Validate the fan settings input."""
    
    # Validate fan output ranges
    if not (0 <= data["min_fan_output"] <= 254):
        raise InvalidRange("Minimum fan output must be between 0 and 254")
    
    if not (1 <= data["max_fan_output"] <= 255):
        raise InvalidRange("Maximum fan output must be between 1 and 255")
    
    # Validate min < max fan output
    if data["min_fan_output"] >= data["max_fan_output"]:
        raise InvalidRange("Minimum fan output must be less than maximum fan output")

    return {"title": "PID Ventilation Control"}


async def validate_basic_input(hass: HomeAssistant, data: dict[str, Any]) -> dict[str, Any]:
    """Validate the fan settings input."""
    
    # Validate fan output ranges
    if not (0 <= data["min_fan_output"] <= 254):
        raise InvalidRange("Minimum fan output must be between 0 and 254")
    
    if not (1 <= data["max_fan_output"] <= 255):
        raise InvalidRange("Maximum fan output must be between 1 and 255")
    
    # Validate min < max fan output
    if data["min_fan_output"] >= data["max_fan_output"]:
        raise InvalidRange("Minimum fan output must be less than maximum fan output")
    
    # Validate remote_device is provided
    if not data.get("remote_device"):
        raise InvalidEntity("Remote device must be selected")

    return {"title": "PID Ventilation Control"}


async def validate_pid_parameters_input(hass: HomeAssistant, data: dict[str, Any]) -> dict[str, Any]:
    """Validate the PID parameters input."""
    
    # Validate setpoint range
    if not (0.0 <= data["setpoint"] <= 5.0):
        raise InvalidRange("Setpoint must be between 0.0 and 5.0")
    
    # Validate Kp range
    if not (0.1 <= data["kp"] <= 1000.0):
        raise InvalidRange("Kp must be between 0.1 and 1000.0")
    
    # Validate update interval range
    if not (10 <= data["update_interval"] <= 3600):
        raise InvalidRange("Update interval must be between 10 and 3600 seconds")

    return {"title": "PID Ventilation Control"}


async def validate_sensor_types_input(hass: HomeAssistant, data: dict[str, Any]) -> dict[str, Any]:
    """Validate the sensor types selection."""
    
    # Check that at least one sensor type is selected
    if not any([data.get("use_co2_sensors"), data.get("use_voc_sensors"), data.get("use_pm_sensors"), data.get("use_humidity_sensors")]):
        raise InvalidSensor("At least one sensor type must be selected")
    
    return {"title": "PID Ventilation Control"}


async def validate_zone_config_input(hass: HomeAssistant, data: dict[str, Any], zone_number: int) -> dict[str, Any]:
    """Validate the zone configuration input and extract device serials."""
    
    # Validate device_from selection
    device_from_id = data.get("device_from")
    if not device_from_id or device_from_id == "":
        raise InvalidZoneConfig(f"Zone {zone_number} 'From' device is required")
    
    # Extract serial number from device_from
    id_from = await get_device_serial_number(hass, device_from_id)
    if not id_from:
        raise InvalidZoneConfig(f"Zone {zone_number} 'From' device serial number could not be extracted")
    
    # Validate device_to selection
    device_to_id = data.get("device_to")
    if not device_to_id or device_to_id == "":
        raise InvalidZoneConfig(f"Zone {zone_number} 'To' device is required")
    
    # Extract serial number from device_to
    id_to = await get_device_serial_number(hass, device_to_id)
    if not id_to:
        raise InvalidZoneConfig(f"Zone {zone_number} 'To' device serial number could not be extracted")
    
    # Validate sensor ID
    sensor_id = data.get("sensor_id", 255)
    if not (0 <= sensor_id <= 255):
        raise InvalidZoneConfig(f"Zone {zone_number} sensor ID must be between 0 and 255")
    
    # Validate fan rate ranges (now in percentages)
    min_fan_pct = data.get("min_fan_rate", 0)
    max_fan_pct = data.get("max_fan_rate", 100)
    
    if not (0 <= min_fan_pct <= 100):
        raise InvalidZoneConfig(f"Zone {zone_number} minimum fan rate must be between 0% and 100%")
    
    if not (0 <= max_fan_pct <= 100):
        raise InvalidZoneConfig(f"Zone {zone_number} maximum fan rate must be between 0% and 100%")
    
    if min_fan_pct >= max_fan_pct:
        raise InvalidZoneConfig(f"Zone {zone_number} minimum fan rate must be less than maximum fan rate")
    
    return {"title": f"Zone {zone_number} Configuration"}


async def validate_sensors_input(hass: HomeAssistant, data: dict[str, Any], sensor_type: str) -> dict[str, Any]:
    """Validate the sensor configuration input for a specific sensor type."""
    
    # Get sensors from the multiple selector
    sensor_key = f"{sensor_type}_sensors"
    sensors = data.get(sensor_key, [])
    
    # Ensure it's a list
    if not isinstance(sensors, list):
        sensors = [sensors] if sensors else []
    
    # Filter out empty values
    sensors = [sensor for sensor in sensors if sensor and str(sensor).strip()]
    
    if not sensors:
        raise InvalidSensor(f"At least one {sensor_type.upper()} sensor must be specified")

    # Validate that all selected sensors exist (optional but helpful)
    for sensor in sensors:
        if sensor and hass.states.get(sensor) is None:
            _LOGGER.warning(f"{sensor_type.upper()} sensor '{sensor}' not found in Home Assistant, but will proceed with setup")

    # Return info that you want to store in the config entry.
    return {
        "title": f"PID Ventilation Control ({len(sensors)} {sensor_type.upper()} sensors)", 
        sensor_key: sensors
    }


async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> dict[str, Any]:
    """Validate the user input allows us to connect."""
    
    # Collect all sensor types
    all_sensors = []
    sensor_types = []
    
    # Check CO2 sensors
    co2_sensors = data.get("co2_sensors", [])
    if not isinstance(co2_sensors, list):
        co2_sensors = [co2_sensors] if co2_sensors else []
    co2_sensors = [sensor for sensor in co2_sensors if sensor and str(sensor).strip()]
    if co2_sensors:
        all_sensors.extend(co2_sensors)
        sensor_types.append("CO2")
    
    # Check VOC sensors
    voc_sensors = data.get("voc_sensors", [])
    if not isinstance(voc_sensors, list):
        voc_sensors = [voc_sensors] if voc_sensors else []
    voc_sensors = [sensor for sensor in voc_sensors if sensor and str(sensor).strip()]
    if voc_sensors:
        all_sensors.extend(voc_sensors)
        sensor_types.append("VOC")
    
    # Check PM sensors
    pm_sensors = data.get("pm_sensors", [])
    if not isinstance(pm_sensors, list):
        pm_sensors = [pm_sensors] if pm_sensors else []
    pm_sensors = [sensor for sensor in pm_sensors if sensor and str(sensor).strip()]
    if pm_sensors:
        all_sensors.extend(pm_sensors)
        sensor_types.append("PM")
    
    # Check Humidity sensors
    humidity_sensors = data.get("humidity_sensors", [])
    if not isinstance(humidity_sensors, list):
        humidity_sensors = [humidity_sensors] if humidity_sensors else []
    humidity_sensors = [sensor for sensor in humidity_sensors if sensor and str(sensor).strip()]
    if humidity_sensors:
        all_sensors.extend(humidity_sensors)
        sensor_types.append("Humidity")
    
    if not all_sensors:
        raise InvalidSensor("At least one sensor must be specified")
    
    # Validate remote_device is provided
    if not data.get("remote_device"):
        raise InvalidEntity("Remote device must be selected")
    
    # Validate min < max fan output
    if data["min_fan_output"] >= data["max_fan_output"]:
        raise InvalidRange("Minimum fan output must be less than maximum fan output")

    # Return info that you want to store in the config entry.
    sensor_types_str = "/".join(sensor_types)
    return {"title": f"PID Ventilation Control ({len(all_sensors)} {sensor_types_str} sensors)", "all_sensors": all_sensors}


def validate_air_quality_index(index_str: str, index_name: str, expected_count: int = 6) -> list[float]:
    """Validate and parse air quality index from comma-separated string."""
    try:
        # Split by comma and strip whitespace
        index_parts = [part.strip() for part in index_str.split(",")]
        
        # Check if we have the expected number of values
        if len(index_parts) != expected_count:
            raise InvalidAirQualityIndex(f"Expected exactly {expected_count} {index_name} values, got {len(index_parts)}")
        
        # Convert to floats and validate
        index_values = []
        for i, part in enumerate(index_parts):
            if not part:
                raise InvalidAirQualityIndex(f"{index_name} value {i+1} is empty")
            
            try:
                index_value = float(part)
            except ValueError:
                raise InvalidAirQualityIndex(f"{index_name} value {i+1} '{part}' is not a valid number")
            
            # Validate that values are in strictly ascending order
            if i > 0 and index_value <= index_values[-1]:
                if index_value == index_values[-1]:
                    raise InvalidAirQualityIndex(f"{index_name} values must be in strictly ascending order. Value {i+1} ({index_value}) is equal to value {i} ({index_values[-1]})")
                else:
                    raise InvalidAirQualityIndex(f"{index_name} values must be in strictly ascending order. Value {i+1} ({index_value}) is less than value {i} ({index_values[-1]})")
            
            # Validate range based on index type
            if index_name.startswith("CO2") and not (0 <= index_value <= 10000):
                raise InvalidAirQualityIndex(f"CO2 values must be between 0 and 10000 ppm")
            elif index_name.startswith("VOC (parts)") and not (0 <= index_value <= 1000):
                raise InvalidAirQualityIndex(f"VOC (parts) values must be between 0 and 1000 ppm")
            elif index_name.startswith("VOC") and not index_name.startswith("VOC (parts)") and not (0 <= index_value <= 10000):
                raise InvalidAirQualityIndex(f"VOC values must be between 0 and 10000 µg/m³")
            elif index_name.startswith("PM") and not (0 <= index_value <= 1000):
                raise InvalidAirQualityIndex(f"PM values must be between 0 and 1000 µg/m³")
            elif index_name.startswith("Humidity") and not (0 <= index_value <= 100):
                raise InvalidAirQualityIndex(f"Humidity values must be between 0 and 100 %")
            
            index_values.append(index_value)
        
        # Final validation: ensure the entire sequence is strictly ascending
        for i in range(1, len(index_values)):
            if index_values[i] <= index_values[i-1]:
                raise InvalidAirQualityIndex(f"{index_name} values must be in strictly ascending order throughout")
        
        # Additional validation: ensure first value is reasonable (not negative for most cases)
        if index_values[0] < 0:
            raise InvalidAirQualityIndex(f"{index_name} values cannot be negative")
        
        return index_values
        
    except Exception as e:
        if isinstance(e, InvalidAirQualityIndex):
            raise
        raise InvalidAirQualityIndex(f"Invalid {index_name} format: {str(e)}")


def validate_ki_times(ki_times_str: str) -> list[int]:
    """Validate and parse Ki times from comma-separated string."""
    try:
        # Split by comma and strip whitespace
        ki_parts = [part.strip() for part in ki_times_str.split(",")]
        
        # Check if we have exactly 6 values
        if len(ki_parts) != 6:
            raise InvalidKiTimes(f"Expected exactly 6 Ki time values, got {len(ki_parts)}")
        
        # Convert to integers and validate range
        ki_times = []
        for i, part in enumerate(ki_parts):
            if not part:
                raise InvalidKiTimes(f"Ki time value {i+1} is empty")
            
            try:
                ki_value = int(part)
            except ValueError:
                raise InvalidKiTimes(f"Ki time value {i+1} '{part}' is not a valid number")
            
            if not (60 <= ki_value <= 7200):
                raise InvalidKiTimes(f"Ki time value {i+1} '{ki_value}' must be between 60 and 7200 seconds")
            
            ki_times.append(ki_value)
        
        return ki_times
        
    except Exception as e:
        if isinstance(e, InvalidKiTimes):
            raise
        raise InvalidKiTimes(f"Invalid Ki times format: {str(e)}")


def validate_sensor_zone_assignment(sensors: list[str], zones_str: str, num_zones: int, sensor_type: str) -> list[int]:
    """Validate and parse zone assignments for sensors."""
    if not zones_str.strip():
        # Default to zone 1 for all sensors
        return [1] * len(sensors)
    
    try:
        # Split by comma and strip whitespace
        zone_parts = [part.strip() for part in zones_str.split(",")]
        
        # Check if we have the same number of zones as sensors
        if len(zone_parts) != len(sensors):
            raise InvalidRange(f"{sensor_type} sensors ({len(sensors)}) and zones ({len(zone_parts)}) count mismatch. Please provide one zone number per sensor.")
        
        # Convert to integers and validate range
        zones = []
        for i, part in enumerate(zone_parts):
            if not part:
                raise InvalidRange(f"Zone assignment {i+1} for {sensor_type} sensor is empty")
            
            try:
                zone_value = int(part)
            except ValueError:
                raise InvalidRange(f"Zone assignment '{part}' for {sensor_type} sensor {i+1} is not a valid number")
            
            if not (1 <= zone_value <= num_zones):
                raise InvalidRange(f"Zone assignment '{zone_value}' for {sensor_type} sensor {i+1} must be between 1 and {num_zones}")
            
            zones.append(zone_value)
        
        return zones
        
    except InvalidRange:
        # Re-raise our custom exception as-is
        raise
    except Exception as e:
        # Catch any other unexpected errors
        raise InvalidRange(f"Invalid {sensor_type} zone assignments: {str(e)}")


class ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for PID Ventilation Control."""

    VERSION = 1

    def __init__(self):
        """Initialize the config flow."""
        self._data = {}
        self._sensor_types = {}
        self._num_zones = 1  # Track number of zones (1 if no valves, else number of valves)
        self._current_zone = 1  # Track which zone is currently being configured
        # Store zone assignments for each sensor type
        self._sensor_zones = {
            "co2_sensors": [],
            "voc_sensors": [],
            "pm_sensors": [],
            "humidity_sensors": [],
        }
        # Track which specific sensor classes were selected
        self._sensor_classes = {
            "co2": False,
            "pm1": False,
            "pm25": False,
            "pm10": False,
            "voc": False,
            "voc_parts": False,
            "humidity": False,
        }

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        """Get the options flow for this handler."""
        return OptionsFlowHandler(config_entry)

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step - device selection."""
        errors: dict[str, str] = {}
        
        # Check if already configured (only allow one instance)
        if self._async_current_entries():
            return self.async_abort(reason="already_configured")
        
        if user_input is not None:
            try:
                # Validate the device selection input
                info = await validate_device_selection_input(self.hass, user_input)
                
                # Store device selection data
                self._data.update(user_input)
                
                # Get number of zones from user selection
                self._num_zones = user_input.get("num_zones", 1)
                if isinstance(self._num_zones, str):
                    self._num_zones = int(self._num_zones)
                
                # Extract remote device serial number for prefilling zone configs
                remote_serial = await get_device_serial_number(self.hass, user_input["remote_device"])
                
                # Store serial number for later use in zone configurations
                self._data["remote_serial"] = remote_serial or ""
                
                _LOGGER.info(f"Device selection: {self._num_zones} zones, Remote: {remote_serial}")
                
                # Initialize zone configurations
                if "zone_configs" not in self._data:
                    self._data["zone_configs"] = {}
                
                # Go to first zone configuration step
                return await self.async_step_zone_config_1()
                
            except InvalidEntity as err:
                # Determine which device field had the error
                error_msg = str(err).lower()
                if "remote" in error_msg:
                    errors["remote_device"] = str(err)
                else:
                    errors["base"] = str(err)
                _LOGGER.warning(f"Invalid entity: {err}")
            except Exception as err:  # pylint: disable=broad-except
                errors["base"] = "unknown"
                _LOGGER.error(f"Unexpected error in device selection: {err}", exc_info=True)

        # Show the device selection form
        return self.async_show_form(
            step_id="user",
            data_schema=get_device_selection_schema(),
            errors=errors,
        )

    async def async_step_zone_config_1(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle zone 1 configuration step."""
        return await self._async_step_zone_config(1, user_input)

    async def async_step_zone_config_2(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle zone 2 configuration step."""
        return await self._async_step_zone_config(2, user_input)

    async def async_step_zone_config_3(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle zone 3 configuration step."""
        return await self._async_step_zone_config(3, user_input)

    async def async_step_zone_config_4(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle zone 4 configuration step."""
        return await self._async_step_zone_config(4, user_input)

    async def async_step_zone_config_5(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle zone 5 configuration step."""
        return await self._async_step_zone_config(5, user_input)

    async def _async_step_zone_config(
        self, zone_number: int, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle zone configuration step for any zone number."""
        errors: dict[str, str] = {}
        
        # First, handle the case where we might have form validation errors but back button was pressed
        if user_input is not None:
            # Check if back button was pressed - this should override any validation errors
            back_pressed = user_input.get("back", False)
            if back_pressed is True or str(back_pressed).lower() == "true":
                if zone_number == 1:
                    return await self.async_step_user()
                else:
                    # Go back to previous zone configuration
                    prev_zone = zone_number - 1
                    return await getattr(self, f"async_step_zone_config_{prev_zone}")()
            
            try:
                # Validate the zone configuration input (back button already handled above)
                _LOGGER.info(f"Validating zone {zone_number} config input: {user_input}")
                info = await validate_zone_config_input(self.hass, user_input, zone_number)
                
                # Extract device serials for storage
                device_from_id = user_input.get("device_from")
                device_to_id = user_input.get("device_to")
                id_from = await get_device_serial_number(self.hass, device_from_id)
                id_to = await get_device_serial_number(self.hass, device_to_id)
                
                # Store zone configuration data (fan rates are now in percentages)
                if "zone_configs" not in self._data:
                    self._data["zone_configs"] = {}
                self._data["zone_configs"][zone_number] = {
                    "device_from": device_from_id,  # Store device ID for config flow
                    "device_to": device_to_id,      # Store device ID for config flow
                    "id_from": id_from,             # Store serial for commands
                    "id_to": id_to,                 # Store serial for commands
                    "sensor_id": user_input["sensor_id"],
                    "min_fan_rate": user_input["min_fan_rate"],  # Now stored as percentage (0-100)
                    "max_fan_rate": user_input["max_fan_rate"],  # Now stored as percentage (0-100)
                }
                
                # Check if we need to configure more zones
                if zone_number < self._num_zones:
                    # Go to next zone configuration
                    next_zone = zone_number + 1
                    return await getattr(self, f"async_step_zone_config_{next_zone}")()
                else:
                    # All zones configured, go to sensor type selection
                    return await self.async_step_sensor_types()
                    
            except InvalidZoneConfig as err:
                # Provide specific error messages based on validation
                error_msg = str(err).lower()
                if "'from' device" in error_msg:
                    errors["device_from"] = str(err)
                elif "'to' device" in error_msg:
                    errors["device_to"] = str(err)
                elif "min_fan" in error_msg:
                    errors["min_fan_rate"] = str(err)
                elif "max_fan" in error_msg:
                    errors["max_fan_rate"] = str(err)
                else:
                    errors["base"] = str(err)
                _LOGGER.warning(f"Invalid zone config: {err}")
            except Exception as err:  # pylint: disable=broad-except
                # Check if this was a back button request that failed
                if user_input and user_input.get("back", False):
                    # Force navigation back regardless of validation errors
                    if zone_number == 1:
                        return await self.async_step_user()
                    else:
                        prev_zone = zone_number - 1
                        return await getattr(self, f"async_step_zone_config_{prev_zone}")()
                
                errors["base"] = "unknown"
                _LOGGER.error(f"Unexpected error in zone config step: {err}", exc_info=True)

        # Show the zone configuration form
        current_config = self._data.get("zone_configs", {}).get(zone_number, {})
        remote_device_id = self._data.get("remote_device")
        
        return self.async_show_form(
            step_id=f"zone_config_{zone_number}",
            data_schema=get_zone_config_schema(zone_number, current_config, remote_device_id),
            errors=errors,
        )

    async def async_step_fan_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the fan settings step."""
        errors: dict[str, str] = {}
        
        if user_input is not None:
            # Check for back button
            if user_input.get("back"):
                return await self.async_step_user()
                
            try:
                # Validate the fan settings input
                info = await validate_fan_settings_input(self.hass, user_input)
                
                # Store fan settings data
                self._data.update(user_input)
                
                # Go to sensor type selection step
                return await self.async_step_sensor_types()
                
            except InvalidRange as err:
                # Provide specific error messages based on validation
                error_msg = str(err).lower()
                if "minimum" in error_msg:
                    errors["min_fan_output"] = str(err)
                elif "maximum" in error_msg:
                    errors["max_fan_output"] = str(err)
                else:
                    errors["min_fan_output"] = str(err)
                _LOGGER.warning(f"Invalid range: {err}")
            except Exception as err:  # pylint: disable=broad-except
                errors["base"] = "unknown"
                _LOGGER.error(f"Unexpected error in fan settings: {err}", exc_info=True)

        # Show the fan settings form
        return self.async_show_form(
            step_id="fan_settings",
            data_schema=get_fan_settings_schema(),
            errors=errors,
        )

    async def async_step_pid_parameters(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the final PID parameters step (includes Ki)."""
        errors: dict[str, str] = {}
        
        if user_input is not None:
            # Check for back button
            if user_input.get("back"):
                return await self.async_step_air_quality_indices()
                
            try:
                # Validate the PID parameters input
                await validate_pid_parameters_input(self.hass, user_input)
                
                # Validate and convert Ki times
                ki_times = validate_ki_times(user_input["ki_times"])
                
                # Store PID parameters and Ki times
                self._data.update({
                    "setpoint": user_input["setpoint"],
                    "kp": user_input["kp"],
                    "update_interval": user_input["update_interval"],
                    "ki_times": ki_times,
                })
                
                # Store sensor type selections
                self._data.update(self._sensor_types)
                
                # Store number of zones
                self._data["num_zones"] = self._num_zones
                
                # Count total sensors for title
                total_sensors = 0
                sensor_types = []
                if self._data.get("co2_sensors"):
                    total_sensors += len(self._data["co2_sensors"])
                    sensor_types.append("CO2")
                if self._data.get("voc_sensors"):
                    total_sensors += len(self._data["voc_sensors"])
                    sensor_types.append("VOC")
                if self._data.get("pm_sensors"):
                    total_sensors += len(self._data["pm_sensors"])
                    sensor_types.append("PM")
                if self._data.get("humidity_sensors"):
                    total_sensors += len(self._data["humidity_sensors"])
                    sensor_types.append("Humidity")

                sensor_types_str = "/".join(sensor_types) if sensor_types else "sensors"
                zones_str = f"{self._num_zones} zone{'s' if self._num_zones > 1 else ''}"
                title = f"PID Ventilation Control ({total_sensors} {sensor_types_str} sensors, {zones_str})"

                return self.async_create_entry(title=title, data=self._data)
            except InvalidRange as err:
                # Provide specific error messages based on validation
                error_msg = str(err).lower()
                if "setpoint" in error_msg:
                    errors["setpoint"] = str(err)
                elif "kp" in error_msg:
                    errors["kp"] = str(err)
                elif "interval" in error_msg:
                    errors["update_interval"] = str(err)
                else:
                    errors["base"] = str(err)
                _LOGGER.warning(f"Invalid range: {err}")
            except InvalidKiTimes as err:
                errors["ki_times"] = str(err)
                _LOGGER.warning(f"Invalid Ki times: {err}")
            except Exception as err:  # pylint: disable=broad-except
                errors["base"] = "unknown"
                _LOGGER.error(f"Unexpected error in PID parameters step: {err}", exc_info=True)

        # Show the PID parameters form
        return self.async_show_form(
            step_id="pid_parameters",
            data_schema=get_pid_parameters_schema(),
            errors=errors,
        )

    async def async_step_sensor_types(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the sensor type selection step."""
        errors: dict[str, str] = {}
        
        if user_input is not None:
            # Check for back button
            if user_input.get("back"):
                # Go back to last zone configuration step
                return await getattr(self, f"async_step_zone_config_{self._num_zones}")()
                
            try:
                # Validate the sensor types selection
                await validate_sensor_types_input(self.hass, user_input)
                
                # Store sensor type selections
                self._sensor_types = user_input
                
                # Determine next step based on selections
                if user_input.get("use_co2_sensors"):
                    return await self.async_step_co2_sensors()
                elif user_input.get("use_voc_sensors"):
                    return await self.async_step_voc_sensors()
                elif user_input.get("use_pm_sensors"):
                    return await self.async_step_pm_sensors()
                elif user_input.get("use_humidity_sensors"):
                    return await self.async_step_humidity_sensors()
                else:
                    # This shouldn't happen due to validation, but just in case
                    return await self.async_step_air_quality_indices()
                    
            except InvalidSensor as err:
                errors["base"] = "no_sensor_types"
                _LOGGER.warning(f"Invalid sensor types: {err}")
            except Exception as err:  # pylint: disable=broad-except
                errors["base"] = "unknown"
                _LOGGER.error(f"Unexpected error in sensor types step: {err}", exc_info=True)

        # Show the sensor type selection form
        return self.async_show_form(
            step_id="sensor_types",
            data_schema=get_sensor_types_schema(),
            errors=errors,
        )

    async def async_step_co2_sensors(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the CO2 sensor selection step."""
        errors: dict[str, str] = {}
        
        if user_input is not None:
            # Check for back button
            if user_input.get("back"):
                return await self.async_step_sensor_types()
            
            try:
                # Validate the sensor input
                info = await validate_sensors_input(self.hass, user_input, "co2")
                
                # Store CO2 sensors
                selected_sensors = user_input["co2_sensors"]
                self._data["co2_sensors"] = selected_sensors
                
                # Handle zone assignments
                if self._num_zones > 1 and selected_sensors:
                    # For multi-zone, check if we have zone data
                    has_zone_data = any(key.startswith("co2_sensor_") and key.endswith("_zone") 
                                      for key in user_input.keys())
                    
                    if has_zone_data:
                        # We have zone assignments - process them
                        zones = []
                        for i in range(len(selected_sensors)):
                            zone_key = f"co2_sensor_{i}_zone"
                            zone_value = user_input.get(zone_key, "1")
                            try:
                                zones.append(int(zone_value))
                            except ValueError:
                                zones.append(1)
                        
                        self._sensor_zones["co2_sensors"] = zones
                        self._data["co2_sensor_zones"] = zones
                    else:
                        # No zone data yet - show zone assignment form
                        current_zones = self._sensor_zones.get("co2_sensors", [])
                        zones_schema = get_co2_sensors_zones_schema(
                            self.hass, selected_sensors, current_zones, self._num_zones
                        )
                        
                        if zones_schema:
                            # Create placeholders with sensor names
                            placeholders = {
                                "sensor_count": str(len(selected_sensors)),
                                "zone_count": str(self._num_zones)
                            }
                            
                            # Add sensor names for translation placeholders
                            for i, sensor in enumerate(selected_sensors):
                                sensor_name = get_entity_display_name(self.hass, sensor)
                                placeholders[f"sensor_{i}_name"] = sensor_name
                            
                            return self.async_show_form(
                                step_id="co2_sensors_zones", 
                                data_schema=zones_schema,
                                description_placeholders=placeholders
                            )
                else:
                    # Single zone or no sensors - assign all to zone 1
                    zones = [1] * len(selected_sensors)
                    self._sensor_zones["co2_sensors"] = zones
                    self._data["co2_sensor_zones"] = zones
                
                # Detect CO2 device classes selected
                self._update_co2_classes(selected_sensors)
                
                # Go to next sensor type or air quality indices step
                if self._sensor_types.get("use_voc_sensors"):
                    return await self.async_step_voc_sensors()
                elif self._sensor_types.get("use_pm_sensors"):
                    return await self.async_step_pm_sensors()
                elif self._sensor_types.get("use_humidity_sensors"):
                    return await self.async_step_humidity_sensors()
                else:
                    return await self.async_step_air_quality_indices()
                    
            except InvalidSensor as err:
                errors["co2_sensors"] = "invalid_sensor"
                _LOGGER.warning(f"Invalid CO2 sensor: {err}")
            except InvalidRange as err:
                errors["co2_sensor_zones"] = str(err)
                _LOGGER.warning(f"Invalid CO2 sensor zones: {err}")
            except Exception as err:  # pylint: disable=broad-except
                errors["base"] = "unknown"
                _LOGGER.error(f"Unexpected error in CO2 sensor step: {err}", exc_info=True)

        # Show the CO2 sensor selection form (phase 1)
        current_sensors = self._data.get("co2_sensors", [])
        return self.async_show_form(
            step_id="co2_sensors",
            data_schema=get_co2_sensors_selection_schema(current_sensors),
            errors=errors,
        )

    async def async_step_co2_sensors_zones(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the CO2 sensor zone assignment step."""
        errors: dict[str, str] = {}
        
        if user_input is not None:
            # Check for back button
            if user_input.get("back"):
                return await self.async_step_co2_sensors()
                
            try:
                # Process zone assignments
                selected_sensors = self._data.get("co2_sensors", [])
                zones = []
                for i in range(len(selected_sensors)):
                    zone_key = f"co2_sensor_{i}_zone"
                    zone_value = user_input.get(zone_key, "1")
                    try:
                        zones.append(int(zone_value))
                    except ValueError:
                        zones.append(1)
                
                # Store zone assignments
                self._sensor_zones["co2_sensors"] = zones
                self._data["co2_sensor_zones"] = zones
                
                # Detect CO2 device classes selected
                self._update_co2_classes(selected_sensors)
                
                # Go to next sensor type or air quality indices step
                if self._sensor_types.get("use_voc_sensors"):
                    return await self.async_step_voc_sensors()
                elif self._sensor_types.get("use_pm_sensors"):
                    return await self.async_step_pm_sensors()
                elif self._sensor_types.get("use_humidity_sensors"):
                    return await self.async_step_humidity_sensors()
                else:
                    return await self.async_step_air_quality_indices()
                    
            except InvalidRange as err:
                errors["co2_sensor_zones"] = str(err)
                _LOGGER.warning(f"Invalid CO2 sensor zones: {err}")
            except Exception as err:  # pylint: disable=broad-except
                errors["base"] = "unknown"
                _LOGGER.error(f"Unexpected error in CO2 zones step: {err}", exc_info=True)

        # Show the zone assignment form
        selected_sensors = self._data.get("co2_sensors", [])
        current_zones = self._sensor_zones.get("co2_sensors", [])
        zones_schema = get_co2_sensors_zones_schema(
            self.hass, selected_sensors, current_zones, self._num_zones
        )
        
        if zones_schema:
            # Create placeholders with sensor names
            placeholders = {
                "sensor_count": str(len(selected_sensors)),
                "zone_count": str(self._num_zones)
            }
            
            # Add sensor names for translation placeholders
            for i, sensor in enumerate(selected_sensors):
                sensor_name = get_entity_display_name(self.hass, sensor)
                placeholders[f"sensor_{i}_name"] = sensor_name
            
            return self.async_show_form(
                step_id="co2_sensors_zones",
                data_schema=zones_schema,
                description_placeholders=placeholders,
                errors=errors,
            )
        
        # Fallback - should not happen
        return await self.async_step_voc_sensors()

    async def async_step_voc_sensors(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the VOC sensor selection step."""
        errors: dict[str, str] = {}
        
        if user_input is not None:
            # Check for back button
            if user_input.get("back"):
                # Go back to CO2 zones step if CO2 was selected, otherwise to sensor_types
                if self._sensor_types.get("use_co2_sensors"):
                    return await self.async_step_co2_sensors_zones()
                else:
                    return await self.async_step_sensor_types()
                
            try:
                # Validate the sensor input
                info = await validate_sensors_input(self.hass, user_input, "voc")
                
                # Store VOC sensors
                selected_sensors = user_input["voc_sensors"]
                self._data["voc_sensors"] = selected_sensors
                
                # Detect VOC device classes selected
                self._update_voc_classes(selected_sensors)
                
                # If multiple zones, go to zone assignment, otherwise assign all to zone 1
                if self._num_zones > 1 and selected_sensors:
                    # Initialize zones for newly selected sensors
                    self._sensor_zones["voc_sensors"] = [1] * len(selected_sensors)
                    return await self.async_step_voc_sensors_zones()
                else:
                    # Single zone or no sensors, assign all to zone 1
                    zones = [1] * len(selected_sensors)
                    self._sensor_zones["voc_sensors"] = zones
                    self._data["voc_sensor_zones"] = zones
                    
                    # Go to next sensor type or air quality indices step
                    if self._sensor_types.get("use_pm_sensors"):
                        return await self.async_step_pm_sensors()
                    elif self._sensor_types.get("use_humidity_sensors"):
                        return await self.async_step_humidity_sensors()
                    else:
                        return await self.async_step_air_quality_indices()
                        
            except InvalidSensor as err:
                errors["voc_sensors"] = "invalid_sensor"
                _LOGGER.warning(f"Invalid VOC sensor: {err}")
            except Exception as err:  # pylint: disable=broad-except
                errors["base"] = "unknown"
                _LOGGER.error(f"Unexpected error in VOC sensor step: {err}", exc_info=True)

        # Show the VOC sensor selection form (phase 1)
        current_sensors = self._data.get("voc_sensors", [])
        return self.async_show_form(
            step_id="voc_sensors",
            data_schema=get_voc_sensors_selection_schema(current_sensors),
            errors=errors,
        )

    async def async_step_voc_sensors_zones(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the VOC sensor zone assignment step."""
        errors: dict[str, str] = {}
        
        if user_input is not None:
            # Check for back button
            if user_input.get("back"):
                return await self.async_step_voc_sensors()
                
            try:
                # Process zone assignments
                selected_sensors = self._data.get("voc_sensors", [])
                zones = []
                for i in range(len(selected_sensors)):
                    zone_key = f"voc_sensor_{i}_zone"
                    zone_value = user_input.get(zone_key, "1")
                    try:
                        zones.append(int(zone_value))
                    except ValueError:
                        zones.append(1)
                
                # Store zone assignments
                self._sensor_zones["voc_sensors"] = zones
                self._data["voc_sensor_zones"] = zones
                
                # Go to next sensor type or air quality indices step
                if self._sensor_types.get("use_pm_sensors"):
                    return await self.async_step_pm_sensors()
                elif self._sensor_types.get("use_humidity_sensors"):
                    return await self.async_step_humidity_sensors()
                else:
                    return await self.async_step_air_quality_indices()
                    
            except InvalidRange as err:
                errors["voc_sensor_zones"] = str(err)
                _LOGGER.warning(f"Invalid VOC sensor zones: {err}")
            except Exception as err:  # pylint: disable=broad-except
                errors["base"] = "unknown"
                _LOGGER.error(f"Unexpected error in VOC zones step: {err}", exc_info=True)

        # Show the zone assignment form
        selected_sensors = self._data.get("voc_sensors", [])
        current_zones = self._sensor_zones.get("voc_sensors", [])
        zones_schema = get_voc_sensors_zones_schema(
            self.hass, selected_sensors, current_zones, self._num_zones
        )
        
        if zones_schema:
            # Create placeholders with sensor names
            placeholders = {
                "sensor_count": str(len(selected_sensors)),
                "zone_count": str(self._num_zones)
            }
            
            # Add sensor names for translation placeholders
            for i, sensor in enumerate(selected_sensors):
                sensor_name = get_entity_display_name(self.hass, sensor)
                placeholders[f"sensor_{i}_name"] = sensor_name
            
            return self.async_show_form(
                step_id="voc_sensors_zones",
                data_schema=zones_schema,
                description_placeholders=placeholders,
                errors=errors,
            )
        
        # Fallback - should not happen
        return await self.async_step_pm_sensors()

    async def async_step_pm_sensors(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the PM sensor selection step."""
        errors: dict[str, str] = {}
        
        if user_input is not None:
            # Check for back button
            if user_input.get("back"):
                # Go back to VOC zones step if VOC was selected, else CO2 zones step if CO2 was selected, else sensor_types
                if self._sensor_types.get("use_voc_sensors"):
                    return await self.async_step_voc_sensors_zones()
                elif self._sensor_types.get("use_co2_sensors"):
                    return await self.async_step_co2_sensors_zones()
                else:
                    return await self.async_step_sensor_types()
                
            try:
                # Validate the sensor input
                info = await validate_sensors_input(self.hass, user_input, "pm")
                
                # Store PM sensors
                selected_sensors = user_input["pm_sensors"]
                self._data["pm_sensors"] = selected_sensors
                
                # Detect PM device classes selected
                self._update_pm_classes(selected_sensors)
                
                # If multiple zones, go to zone assignment, otherwise assign all to zone 1
                if self._num_zones > 1 and selected_sensors:
                    # Initialize zones for newly selected sensors
                    self._sensor_zones["pm_sensors"] = [1] * len(selected_sensors)
                    return await self.async_step_pm_sensors_zones()
                else:
                    # Single zone or no sensors, assign all to zone 1
                    zones = [1] * len(selected_sensors)
                    self._sensor_zones["pm_sensors"] = zones
                    self._data["pm_sensor_zones"] = zones
                    
                    # Go to next sensor type or air quality indices step
                    if self._sensor_types.get("use_humidity_sensors"):
                        return await self.async_step_humidity_sensors()
                    else:
                        return await self.async_step_air_quality_indices()
                        
            except InvalidSensor as err:
                errors["pm_sensors"] = "invalid_sensor"
                _LOGGER.warning(f"Invalid PM sensor: {err}")
            except Exception as err:  # pylint: disable=broad-except
                errors["base"] = "unknown"
                _LOGGER.error(f"Unexpected error in PM sensor step: {err}", exc_info=True)

        # Show the PM sensor selection form (phase 1)
        current_sensors = self._data.get("pm_sensors", [])
        return self.async_show_form(
            step_id="pm_sensors",
            data_schema=get_pm_sensors_selection_schema(current_sensors),
            errors=errors,
        )

    async def async_step_pm_sensors_zones(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the PM sensor zone assignment step."""
        errors: dict[str, str] = {}
        
        if user_input is not None:
            # Check for back button
            if user_input.get("back"):
                return await self.async_step_pm_sensors()
                
            try:
                # Process zone assignments
                selected_sensors = self._data.get("pm_sensors", [])
                zones = []
                for i in range(len(selected_sensors)):
                    zone_key = f"pm_sensor_{i}_zone"
                    zone_value = user_input.get(zone_key, "1")
                    try:
                        zones.append(int(zone_value))
                    except ValueError:
                        zones.append(1)
                
                # Store zone assignments
                self._sensor_zones["pm_sensors"] = zones
                self._data["pm_sensor_zones"] = zones
                
                # Go to next sensor type or air quality indices step
                if self._sensor_types.get("use_humidity_sensors"):
                    return await self.async_step_humidity_sensors()
                else:
                    return await self.async_step_air_quality_indices()
                    
            except InvalidRange as err:
                errors["pm_sensor_zones"] = str(err)
                _LOGGER.warning(f"Invalid PM sensor zones: {err}")
            except Exception as err:  # pylint: disable=broad-except
                errors["base"] = "unknown"
                _LOGGER.error(f"Unexpected error in PM zones step: {err}", exc_info=True)

        # Show the zone assignment form
        selected_sensors = self._data.get("pm_sensors", [])
        current_zones = self._sensor_zones.get("pm_sensors", [])
        zones_schema = get_pm_sensors_zones_schema(
            self.hass, selected_sensors, current_zones, self._num_zones
        )
        
        if zones_schema:
            # Create placeholders with sensor names
            placeholders = {
                "sensor_count": str(len(selected_sensors)),
                "zone_count": str(self._num_zones)
            }
            
            # Add sensor names for translation placeholders
            for i, sensor in enumerate(selected_sensors):
                sensor_name = get_entity_display_name(self.hass, sensor)
                placeholders[f"sensor_{i}_name"] = sensor_name
            
            return self.async_show_form(
                step_id="pm_sensors_zones",
                data_schema=zones_schema,
                description_placeholders=placeholders,
                errors=errors,
            )
        
        # Fallback - should not happen
        return await self.async_step_humidity_sensors()

    async def async_step_humidity_sensors(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the Humidity sensor selection step."""
        errors: dict[str, str] = {}
        
        if user_input is not None:
            # Check for back button
            if user_input.get("back"):
                # Go back to PM zones step if PM was selected, else VOC zones step if VOC was selected, else CO2 zones step if CO2 was selected, else sensor_types
                if self._sensor_types.get("use_pm_sensors"):
                    return await self.async_step_pm_sensors_zones()
                elif self._sensor_types.get("use_voc_sensors"):
                    return await self.async_step_voc_sensors_zones()
                elif self._sensor_types.get("use_co2_sensors"):
                    return await self.async_step_co2_sensors_zones()
                else:
                    return await self.async_step_sensor_types()
                
            try:
                # Validate the sensor input
                info = await validate_sensors_input(self.hass, user_input, "humidity")
                
                # Store Humidity sensors
                selected_sensors = user_input["humidity_sensors"]
                self._data["humidity_sensors"] = selected_sensors
                
                # Detect Humidity device classes selected
                self._update_humidity_classes(selected_sensors)
                
                # If multiple zones, go to zone assignment, otherwise assign all to zone 1
                if self._num_zones > 1 and selected_sensors:
                    # Initialize zones for newly selected sensors
                    self._sensor_zones["humidity_sensors"] = [1] * len(selected_sensors)
                    return await self.async_step_humidity_sensors_zones()
                else:
                    # Single zone or no sensors, assign all to zone 1
                    zones = [1] * len(selected_sensors)
                    self._sensor_zones["humidity_sensors"] = zones
                    self._data["humidity_sensor_zones"] = zones
                    
                    # Go to air quality indices step
                    return await self.async_step_air_quality_indices()
                    
            except InvalidSensor as err:
                errors["humidity_sensors"] = "invalid_sensor"
                _LOGGER.warning(f"Invalid Humidity sensor: {err}")
            except Exception as err:  # pylint: disable=broad-except
                errors["base"] = "unknown"
                _LOGGER.error(f"Unexpected error in Humidity sensor step: {err}", exc_info=True)

        # Show the Humidity sensor selection form (phase 1)
        current_sensors = self._data.get("humidity_sensors", [])
        return self.async_show_form(
            step_id="humidity_sensors",
            data_schema=get_humidity_sensors_selection_schema(current_sensors),
            errors=errors,
        )

    async def async_step_humidity_sensors_zones(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the Humidity sensor zone assignment step."""
        errors: dict[str, str] = {}
        
        if user_input is not None:
            # Check for back button
            if user_input.get("back"):
                return await self.async_step_humidity_sensors()
                
            try:
                # Process zone assignments
                selected_sensors = self._data.get("humidity_sensors", [])
                zones = []
                for i in range(len(selected_sensors)):
                    zone_key = f"humidity_sensor_{i}_zone"
                    zone_value = user_input.get(zone_key, "1")
                    try:
                        zones.append(int(zone_value))
                    except ValueError:
                        zones.append(1)
                
                # Store zone assignments
                self._sensor_zones["humidity_sensors"] = zones
                self._data["humidity_sensor_zones"] = zones
                
                # Go to air quality indices step
                return await self.async_step_air_quality_indices()
                    
            except InvalidRange as err:
                errors["humidity_sensor_zones"] = str(err)
                _LOGGER.warning(f"Invalid Humidity sensor zones: {err}")
            except Exception as err:  # pylint: disable=broad-except
                errors["base"] = "unknown"
                _LOGGER.error(f"Unexpected error in Humidity zones step: {err}", exc_info=True)

        # Show the zone assignment form
        selected_sensors = self._data.get("humidity_sensors", [])
        current_zones = self._sensor_zones.get("humidity_sensors", [])
        zones_schema = get_humidity_sensors_zones_schema(
            self.hass, selected_sensors, current_zones, self._num_zones
        )
        
        if zones_schema:
            # Create placeholders with sensor names
            placeholders = {
                "sensor_count": str(len(selected_sensors)),
                "zone_count": str(self._num_zones)
            }
            
            # Add sensor names for translation placeholders
            for i, sensor in enumerate(selected_sensors):
                sensor_name = get_entity_display_name(self.hass, sensor)
                placeholders[f"sensor_{i}_name"] = sensor_name
            
            return self.async_show_form(
                step_id="humidity_sensors_zones",
                data_schema=zones_schema,
                description_placeholders=placeholders,
                errors=errors,
            )
        
        # Fallback - should not happen
        return await self.async_step_air_quality_indices()

    async def async_step_air_quality_indices(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the air quality indices configuration step."""
        errors: dict[str, str] = {}
        
        if user_input is not None:
            # Check for back button
            if user_input.get("back"):
                # Go back to the last sensor type step that was configured
                if self._sensor_types.get("use_humidity_sensors"):
                    return await self.async_step_humidity_sensors()
                elif self._sensor_types.get("use_pm_sensors"):
                    return await self.async_step_pm_sensors()
                elif self._sensor_types.get("use_voc_sensors"):
                    return await self.async_step_voc_sensors()
                elif self._sensor_types.get("use_co2_sensors"):
                    return await self.async_step_co2_sensors()
                else:
                    return await self.async_step_sensor_types()
                
            # Validate each present field separately
            valid = True
            try:
                if self._sensor_types.get("use_co2_sensors"):
                    co2_index = validate_air_quality_index(user_input["co2_index"], "CO2")
                    self._data["co2_index"] = co2_index
            except InvalidAirQualityIndex as e:
                errors["co2_index"] = str(e)
                valid = False
            
            try:
                if self._sensor_types.get("use_voc_sensors") and self._sensor_classes["voc"]:
                    voc_index = validate_air_quality_index(user_input["voc_index"], "VOC")
                    self._data["voc_index"] = voc_index
            except InvalidAirQualityIndex as e:
                errors["voc_index"] = str(e)
                valid = False
            
            try:
                if self._sensor_types.get("use_voc_sensors") and self._sensor_classes["voc_parts"]:
                    voc_ppm_index = validate_air_quality_index(user_input["voc_ppm_index"], "VOC (parts)")
                    self._data["voc_ppm_index"] = voc_ppm_index
            except InvalidAirQualityIndex as e:
                errors["voc_ppm_index"] = str(e)
                valid = False
            
            try:
                if self._sensor_types.get("use_pm_sensors") and self._sensor_classes["pm1"]:
                    pm_1_0_index = validate_air_quality_index(user_input["pm_1_0_index"], "PM1.0")
                    self._data["pm_1_0_index"] = pm_1_0_index
            except InvalidAirQualityIndex as e:
                errors["pm_1_0_index"] = str(e)
                valid = False
            
            try:
                if self._sensor_types.get("use_pm_sensors") and self._sensor_classes["pm25"]:
                    pm_2_5_index = validate_air_quality_index(user_input["pm_2_5_index"], "PM2.5")
                    self._data["pm_2_5_index"] = pm_2_5_index
            except InvalidAirQualityIndex as e:
                errors["pm_2_5_index"] = str(e)
                valid = False
            
            try:
                if self._sensor_types.get("use_pm_sensors") and self._sensor_classes["pm10"]:
                    pm_10_index = validate_air_quality_index(user_input["pm_10_index"], "PM10", expected_count=5)
                    self._data["pm_10_index"] = pm_10_index
            except InvalidAirQualityIndex as e:
                errors["pm_10_index"] = str(e)
                valid = False
            
            try:
                if self._sensor_types.get("use_humidity_sensors") and self._sensor_classes["humidity"]:
                    humidity_index = validate_air_quality_index(user_input["humidity_index"], "Humidity")
                    self._data["humidity_index"] = humidity_index
            except InvalidAirQualityIndex as e:
                errors["humidity_index"] = str(e)
                valid = False

            if valid:
                # Store sensor type selections in data
                self._data.update(self._sensor_types)
                # Go to final PID parameters step
                return await self.async_step_pid_parameters()
        
        # Show the air quality indices form
        return self.async_show_form(
            step_id="air_quality_indices",
            data_schema=self._build_air_quality_indices_schema(),
            errors=errors,
        )

    def _build_air_quality_indices_schema(self) -> vol.Schema:
        """Build AQI schema only for selected sensor classes."""
        schema_dict = {}
        if self._sensor_types.get("use_co2_sensors"):
            schema_dict[vol.Required("co2_index", default=",".join(map(str, DEFAULT_CO2_INDEX)))] = cv.string
        if self._sensor_types.get("use_voc_sensors") and self._sensor_classes["voc"]:
            schema_dict[vol.Required("voc_index", default=",".join(map(str, DEFAULT_VOC_INDEX)))] = cv.string
        if self._sensor_types.get("use_voc_sensors") and self._sensor_classes["voc_parts"]:
            schema_dict[vol.Required("voc_ppm_index", default=",".join(map(str, DEFAULT_VOC_PPM_INDEX)))] = cv.string
        if self._sensor_types.get("use_pm_sensors") and self._sensor_classes["pm1"]:
            schema_dict[vol.Required("pm_1_0_index", default=",".join(map(str, DEFAULT_PM_1_0_INDEX)))] = cv.string
        if self._sensor_types.get("use_pm_sensors") and self._sensor_classes["pm25"]:
            schema_dict[vol.Required("pm_2_5_index", default=",".join(map(str, DEFAULT_PM_2_5_INDEX)))] = cv.string
        if self._sensor_types.get("use_pm_sensors") and self._sensor_classes["pm10"]:
            schema_dict[vol.Required("pm_10_index", default=",".join(map(str, DEFAULT_PM_10_INDEX)))] = cv.string
        if self._sensor_types.get("use_humidity_sensors") and self._sensor_classes["humidity"]:
            schema_dict[vol.Required("humidity_index", default=",".join(map(str, DEFAULT_HUMIDITY_INDEX)))] = cv.string
        schema_dict[vol.Optional("back", default=False)] = selector.BooleanSelector()
        return vol.Schema(schema_dict, extra=vol.ALLOW_EXTRA)

    def _detect_device_classes(self, entity_ids: list[str]) -> set[str]:
        """Detect device_class values for given entity IDs using registry/state."""
        classes: set[str] = set()
        try:
            registry = er.async_get(self.hass)
        except Exception:  # registry may not be available early
            registry = None
        for eid in entity_ids or []:
            dc = None
            if registry:
                entry = registry.async_get(eid)
                if entry and getattr(entry, "device_class", None):
                    dc = entry.device_class
            if dc is None:
                state = self.hass.states.get(eid)
                if state:
                    dc = state.attributes.get("device_class")
            if isinstance(dc, str):
                classes.add(dc)
        return classes

    def _update_co2_classes(self, entity_ids: list[str]) -> None:
        classes = self._detect_device_classes(entity_ids)
        self._sensor_classes["co2"] = "carbon_dioxide" in classes

    def _update_voc_classes(self, entity_ids: list[str]) -> None:
        classes = self._detect_device_classes(entity_ids)
        self._sensor_classes["voc"] = "volatile_organic_compounds" in classes
        self._sensor_classes["voc_parts"] = "volatile_organic_compounds_parts" in classes

    def _update_pm_classes(self, entity_ids: list[str]) -> None:
        classes = self._detect_device_classes(entity_ids)
        self._sensor_classes["pm1"] = "pm1" in classes
        self._sensor_classes["pm25"] = "pm25" in classes
        self._sensor_classes["pm10"] = "pm10" in classes

    def _update_humidity_classes(self, entity_ids: list[str]) -> None:
        classes = self._detect_device_classes(entity_ids)
        self._sensor_classes["humidity"] = "humidity" in classes

class InvalidSensor(HomeAssistantError):
    """Error to indicate sensor not found."""


class InvalidEntity(HomeAssistantError):
    """Error to indicate entity not found."""


class InvalidRange(HomeAssistantError):
    """Error to indicate invalid range."""


class InvalidKiTimes(HomeAssistantError):
    """Error to indicate invalid Ki times format."""


class InvalidAirQualityIndex(HomeAssistantError):
    """Error to indicate invalid air quality index format."""


class InvalidZoneConfig(HomeAssistantError):
    """Error to indicate invalid zone configuration."""


class OptionsFlowHandler(config_entries.OptionsFlow):
    """Handle options flow for PID Ventilation Control."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        # Use improved merge but don't allow cleanup during initialization to preserve user data
        self._data = merge_config_with_validation(config_entry.data, config_entry.options, allow_cleanup=False)
        # Extract validated zone count from the properly merged config
        zone_configs = self._data.get("zone_configs", {})
        self._validated_num_zones = len(zone_configs) if zone_configs else int(self._data.get("num_zones", 1))
        _LOGGER.debug(f"Options flow initialized with {self._validated_num_zones} validated zones (preserving existing configs)")

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Top-level options menu."""
        # Use validated zone count to prevent menu inconsistencies
        num_zones = self._validated_num_zones
        menu_options = [
            "device_configuration",
        ]
        
        # Add zone configuration options based on validated count
        for zone_num in range(1, num_zones + 1):
            menu_options.append(f"zone_{zone_num}_config")
        
        # Add sensor type selection (always available)
        menu_options.append("sensor_types")
        
        # Add direct sensor configuration options for enabled sensor types
        if self._data.get("use_co2_sensors") or self._data.get("co2_sensors"):
            menu_options.append("co2_sensors_options")
        if self._data.get("use_voc_sensors") or self._data.get("voc_sensors"):
            menu_options.append("voc_sensors_options")
        if self._data.get("use_pm_sensors") or self._data.get("pm_sensors"):
            menu_options.append("pm_sensors_options")
        if self._data.get("use_humidity_sensors") or self._data.get("humidity_sensors"):
            menu_options.append("humidity_sensors_options")
        
        # Add other settings
        menu_options.extend([
            "iaq_settings", 
            "pid_settings",
        ])
        
        return self.async_show_menu(
            step_id="init",
            menu_options=menu_options,
        )

    # Fan settings -----------------------------------------------------------
    def _schema_fan(self) -> vol.Schema:
        cur = self._data
        return vol.Schema({
            vol.Required("min_fan_output", default=cur.get("min_fan_output", 0)): vol.Coerce(int),
            vol.Required("max_fan_output", default=cur.get("max_fan_output", 255)): vol.Coerce(int),
            vol.Required("remote_device", default=cur.get("remote_device")): selector.DeviceSelector(
                selector.DeviceSelectorConfig(integration="ramses_cc")
            ),
        }, extra=vol.ALLOW_EXTRA)

    async def async_step_fan_settings(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        errors: dict[str, str] = {}
        if user_input is not None:
            try:
                # Validate each field individually to provide specific error messages
                validated_data = {}
                
                # Validate min_fan_output
                try:
                    if not (0 <= user_input["min_fan_output"] <= 254):
                        raise InvalidRange("Minimum fan output must be between 0 and 254")
                    validated_data["min_fan_output"] = user_input["min_fan_output"]
                except (ValueError, InvalidRange) as e:
                    errors["min_fan_output"] = str(e)
                
                # Validate max_fan_output
                try:
                    if not (1 <= user_input["max_fan_output"] <= 255):
                        raise InvalidRange("Maximum fan output must be between 1 and 255")
                    validated_data["max_fan_output"] = user_input["max_fan_output"]
                except (ValueError, InvalidRange) as e:
                    errors["max_fan_output"] = str(e)
                
                # Validate min < max only if both are valid
                if "min_fan_output" not in errors and "max_fan_output" not in errors:
                    if user_input["min_fan_output"] >= user_input["max_fan_output"]:
                        errors["min_fan_output"] = "Minimum fan output must be less than maximum fan output"
                
                # Validate remote_device
                try:
                    if not user_input.get("remote_device"):
                        raise InvalidEntity("Remote device must be selected")
                    validated_data["remote_device"] = user_input["remote_device"]
                except InvalidEntity as e:
                    errors["remote_device"] = str(e)
                
                # If no validation errors, create the entry
                if not errors:
                    options = {**self.config_entry.options}
                    options.update(validated_data)
                    return self.async_create_entry(title="", data=options)
                    
            except Exception as e:
                _LOGGER.error(f"Unknown error in fan_settings: {e}", exc_info=True)
                errors["base"] = "unknown"
        return self.async_show_form(step_id="fan_settings", data_schema=self._schema_fan(), errors=errors)

    # IAQ settings (indices) -------------------------------------------------
    def _schema_iaq(self) -> vol.Schema:
        cur = self._data
        return vol.Schema({
            vol.Optional("co2_index", default=",".join(map(str, cur.get("co2_index", DEFAULT_CO2_INDEX)))): cv.string,
            vol.Optional("voc_index", default=",".join(map(str, cur.get("voc_index", DEFAULT_VOC_INDEX)))): cv.string,
            vol.Optional("voc_ppm_index", default=",".join(map(str, cur.get("voc_ppm_index", DEFAULT_VOC_PPM_INDEX)))): cv.string,
            vol.Optional("pm_1_0_index", default=",".join(map(str, cur.get("pm_1_0_index", DEFAULT_PM_1_0_INDEX)))): cv.string,
            vol.Optional("pm_2_5_index", default=",".join(map(str, cur.get("pm_2_5_index", DEFAULT_PM_2_5_INDEX)))): cv.string,
            vol.Optional("pm_10_index", default=",".join(map(str, cur.get("pm_10_index", DEFAULT_PM_10_INDEX)))): cv.string,
            vol.Optional("humidity_index", default=",".join(map(str, cur.get("humidity_index", DEFAULT_HUMIDITY_INDEX)))): cv.string,
        }, extra=vol.ALLOW_EXTRA)

    async def async_step_iaq_settings(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        errors: dict[str, str] = {}
        if user_input is not None:
            try:
                # Validate each field individually to provide specific error messages
                for key, name in (
                    ("co2_index", "CO2"),
                    ("voc_index", "VOC"),
                    ("voc_ppm_index", "VOC (parts)"),
                    ("pm_1_0_index", "PM1.0"),
                    ("pm_2_5_index", "PM2.5"),
                    ("pm_10_index", "PM10"),
                    ("humidity_index", "Humidity"),
                ):
                    if key in user_input:
                        try:
                            self._data[key] = validate_air_quality_index(user_input[key], name)
                        except InvalidAirQualityIndex as e:
                            # Show the specific error message for this field
                            errors[key] = str(e)
                
                # If no validation errors, create the entry
                if not errors:
                    return self.async_create_entry(title="", data={**self.config_entry.options, **{key: self._data[key] for key in self._data if key.endswith('_index')}})
                    
            except Exception as e:
                _LOGGER.error(f"Unknown error in iaq_settings: {e}", exc_info=True)
                errors["base"] = "unknown"
        return self.async_show_form(step_id="iaq_settings", data_schema=self._schema_iaq(), errors=errors)

    # PID settings -----------------------------------------------------------
    def _schema_pid(self) -> vol.Schema:
        cur = self._data
        
        # Try to get the current runtime value directly from hass.data
        current_setpoint = cur.get("setpoint", 0.5)  # Default fallback
        try:
            # Get the live runtime value directly from hass.data (same source as the number entity)
            domain_data = self.hass.data.get(DOMAIN, {})
            entry_data = domain_data.get(self.config_entry.entry_id, {})
            if isinstance(entry_data, dict):
                runtime_setpoint = entry_data.get("current_setpoint")
                if runtime_setpoint is not None:
                    _LOGGER.info(f"Config flow schema: using runtime setpoint {runtime_setpoint} (config was {current_setpoint})")
                    current_setpoint = runtime_setpoint
                else:
                    _LOGGER.info(f"Config flow schema: no runtime setpoint, using config setpoint {current_setpoint}")
            else:
                _LOGGER.warning(f"Config flow schema: invalid entry_data type {type(entry_data)}")
        except Exception as e:
            _LOGGER.warning(f"Failed to get runtime setpoint, using config value: {e}")
            # If we can't get the runtime value, stick with the configured value
            pass
        
        # Custom validator for locale-aware float input with range validation
        def validate_float_with_locale(min_val: float, max_val: float):
            return vol.All(
                vol.Coerce(str),
                lambda v: str(v).replace(',', '.'),  # Convert comma to dot for locale compatibility
                vol.Coerce(float),
                vol.Range(min=min_val, max=max_val),
            )
        
        return vol.Schema({
            vol.Required("setpoint", default=current_setpoint): validate_float_with_locale(0.0, 5.0),
            vol.Required("kp", default=cur.get("kp", 25.5)): validate_float_with_locale(0.1, 1000.0),
            vol.Required("ki_times", default=",".join(map(str, cur.get("ki_times", [3600, 1800, 900, 450, 225, 150])))): cv.string,
            vol.Required("update_interval", default=cur.get("update_interval", 300)): vol.Coerce(int),
        }, extra=vol.ALLOW_EXTRA)

    async def async_step_pid_settings(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        errors: dict[str, str] = {}
        if user_input is not None:
            try:
                # Validate each field individually to provide specific error messages
                validated_data = {}
                
                # Validate setpoint
                try:
                    if not (0.0 <= user_input["setpoint"] <= 5.0):
                        raise InvalidRange("Setpoint must be between 0.0 and 5.0")
                    validated_data["setpoint"] = user_input["setpoint"]
                except (ValueError, InvalidRange) as e:
                    errors["setpoint"] = str(e)
                
                # Validate Kp
                try:
                    if not (0.1 <= user_input["kp"] <= 1000.0):
                        raise InvalidRange("Kp must be between 0.1 and 1000.0")
                    validated_data["kp"] = user_input["kp"]
                except (ValueError, InvalidRange) as e:
                    errors["kp"] = str(e)
                
                # Validate update_interval
                try:
                    if not (10 <= user_input["update_interval"] <= 3600):
                        raise InvalidRange("Update interval must be between 10 and 3600 seconds")
                    validated_data["update_interval"] = user_input["update_interval"]
                except (ValueError, InvalidRange) as e:
                    errors["update_interval"] = str(e)
                
                # Validate ki_times
                try:
                    validated_data["ki_times"] = validate_ki_times(user_input["ki_times"])
                except InvalidKiTimes as e:
                    errors["ki_times"] = str(e)
                
                # If no validation errors, create the entry
                if not errors:
                    # Update runtime setpoint to match config to keep them in sync
                    domain_data = self.hass.data.get(DOMAIN, {})
                    entry_data = domain_data.get(self.config_entry.entry_id, {})
                    if isinstance(entry_data, dict) and "setpoint" in validated_data:
                        old_setpoint = entry_data.get("current_setpoint")
                        entry_data["current_setpoint"] = validated_data["setpoint"]
                        _LOGGER.info(f"Config flow updated runtime setpoint: {old_setpoint} → {validated_data['setpoint']}")
                        
                        # Schedule an async task to update the number entity after the config is saved
                        async def update_number_entity():
                            try:
                                # Small delay to ensure the config entry is updated first
                                import asyncio
                                await asyncio.sleep(0.1)
                                
                                # Find the setpoint number entity and schedule a state update
                                entity_registry = er.async_get(self.hass)
                                unique_id = f"{self.config_entry.entry_id}_setpoint"
                                entity_id = entity_registry.async_get_entity_id("number", DOMAIN, unique_id)
                                
                                if entity_id:
                                    # Get the entity object and trigger an update
                                    state = self.hass.states.get(entity_id)
                                    if state:
                                        # Update the state to reflect the new value
                                        self.hass.states.async_set(
                                            entity_id, 
                                            validated_data["setpoint"], 
                                            state.attributes
                                        )
                                        _LOGGER.debug(f"Updated number entity {entity_id} state to {validated_data['setpoint']}")
                                    else:
                                        _LOGGER.warning(f"Could not find state for entity {entity_id}")
                                else:
                                    _LOGGER.warning(f"Could not find entity with unique_id {unique_id}")
                            except Exception as e:
                                _LOGGER.error(f"Error updating number entity: {e}")
                        
                        # Schedule the update task
                        self.hass.async_create_task(update_number_entity())
                    else:
                        _LOGGER.warning(f"Could not update runtime setpoint - entry_data: {type(entry_data)}, has setpoint: {'setpoint' in validated_data}")
                    
                    return self.async_create_entry(title="", data={**self.config_entry.options, **validated_data})
                    
            except Exception as e:
                _LOGGER.error(f"Unknown error in pid_settings: {e}", exc_info=True)
                errors["base"] = "unknown"
        return self.async_show_form(step_id="pid_settings", data_schema=self._schema_pid(), errors=errors)

    # Sensor type selection options -------------------------------------------
    async def async_step_sensor_types(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle sensor type selection in options flow."""
        errors: dict[str, str] = {}
        
        if user_input is not None:
            try:
                # Store sensor type selections
                updated_data = {}
                for sensor_type in ("use_co2_sensors", "use_voc_sensors", "use_pm_sensors", "use_humidity_sensors"):
                    updated_data[sensor_type] = user_input.get(sensor_type, False)
                
                # Validate that at least one sensor type is selected
                if not any(updated_data.values()):
                    raise InvalidSensor("At least one sensor type must be selected")
                
                # Get current state to compare changes
                old_data = {**self.config_entry.data, **self.config_entry.options}
                
                # Handle deselected sensor types - clear their data
                for sensor_type, enabled in updated_data.items():
                    if not enabled and old_data.get(sensor_type, False):
                        # This sensor type was deselected, clear its data
                        sensor_name = sensor_type.replace("use_", "").replace("_sensors", "")
                        self._data[f"{sensor_name}_sensors"] = []
                        self._data[f"{sensor_name}_sensor_zones"] = []
                
                # Store the selections
                self._data.update(updated_data)
                
                # Determine which sensor types were newly enabled (changed from False to True)
                newly_enabled = []
                for sensor_type, enabled in updated_data.items():
                    if enabled and not old_data.get(sensor_type, False):
                        newly_enabled.append(sensor_type)
                
                # If nothing new was enabled (only deselections or no changes), save and exit
                if not newly_enabled:
                    sensor_data = {}
                    for key in ("use_co2_sensors", "use_voc_sensors", "use_pm_sensors", "use_humidity_sensors",
                               "co2_sensors", "voc_sensors", "pm_sensors", "humidity_sensors",
                               "co2_sensor_zones", "voc_sensor_zones", "pm_sensor_zones", "humidity_sensor_zones"):
                        if key in self._data:
                            sensor_data[key] = self._data[key]
                    return self.async_create_entry(title="", data={**self.config_entry.options, **sensor_data})
                
                # Navigate to the first newly enabled sensor type
                for sensor_type in ("use_co2_sensors", "use_voc_sensors", "use_pm_sensors", "use_humidity_sensors"):
                    if sensor_type in newly_enabled:
                        if sensor_type == "use_co2_sensors":
                            return await self.async_step_co2_sensors_options()
                        elif sensor_type == "use_voc_sensors":
                            return await self.async_step_voc_sensors_options()
                        elif sensor_type == "use_pm_sensors":
                            return await self.async_step_pm_sensors_options()
                        elif sensor_type == "use_humidity_sensors":
                            return await self.async_step_humidity_sensors_options()
                
                # Fallback - should not reach here
                return self.async_create_entry(title="", data={**self.config_entry.options, **updated_data})
                    
            except InvalidSensor as err:
                errors["base"] = "no_sensor_types"
                _LOGGER.warning(f"Invalid sensor types: {err}")
            except Exception as err:
                errors["base"] = "unknown"
                _LOGGER.error(f"Unexpected error in sensor types options: {err}", exc_info=True)

        # Show the sensor type selection form
        return self.async_show_form(
            step_id="sensor_types",
            data_schema=self._schema_sensor_types(),
            errors=errors,
        )

    def _schema_sensor_types(self) -> vol.Schema:
        """Generate schema for sensor type selection with current values as defaults."""
        cur = self._data
        return vol.Schema({
            vol.Optional("use_co2_sensors", default=cur.get("use_co2_sensors", bool(cur.get("co2_sensors")))): cv.boolean,
            vol.Optional("use_voc_sensors", default=cur.get("use_voc_sensors", bool(cur.get("voc_sensors")))): cv.boolean,
            vol.Optional("use_pm_sensors", default=cur.get("use_pm_sensors", bool(cur.get("pm_sensors")))): cv.boolean,
            vol.Optional("use_humidity_sensors", default=cur.get("use_humidity_sensors", bool(cur.get("humidity_sensors")))): cv.boolean,
        }, extra=vol.ALLOW_EXTRA)

    # CO2 sensor options -------------------------------------------------------
    async def async_step_co2_sensors_options(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle CO2 sensor selection in options flow."""
        errors: dict[str, str] = {}
        
        if user_input is not None:
            try:
                # Store CO2 sensors
                sensors = user_input.get("co2_sensors", [])
                if not isinstance(sensors, list):
                    sensors = [sensors] if sensors else []
                sensors = [s for s in sensors if s]
                
                # Enable CO2 sensors if any are selected
                if sensors:
                    self._data["use_co2_sensors"] = True
                    self._data["co2_sensors"] = sensors
                else:
                    # If no sensors selected, disable CO2 sensors
                    self._data["use_co2_sensors"] = False
                    self._data["co2_sensors"] = []
                
                # Handle zone assignments if multiple zones and sensors exist
                num_zones = self._data.get("num_zones", 1)
                if num_zones > 1 and sensors:
                    return await self.async_step_co2_sensors_zones_options()
                else:
                    # Single zone or no sensors, assign all to zone 1
                    self._data["co2_sensor_zones"] = [1] * len(sensors)
                    
                    # Check if there are other newly enabled sensor types to configure
                    old_data = {**self.config_entry.data, **self.config_entry.options}
                    remaining_new_types = []
                    for sensor_type in ("use_voc_sensors", "use_pm_sensors", "use_humidity_sensors"):
                        if (self._data.get(sensor_type) and not old_data.get(sensor_type, False)):
                            remaining_new_types.append(sensor_type)
                    
                    if remaining_new_types:
                        # Continue to next newly enabled sensor type
                        return await self._next_sensor_step_options("co2")
                    else:
                        # No more new sensor types, save and return to menu
                        sensor_data = {}
                        for key in ("use_co2_sensors", "use_voc_sensors", "use_pm_sensors", "use_humidity_sensors",
                                   "co2_sensors", "voc_sensors", "pm_sensors", "humidity_sensors",
                                   "co2_sensor_zones", "voc_sensor_zones", "pm_sensor_zones", "humidity_sensor_zones"):
                            if key in self._data:
                                sensor_data[key] = self._data[key]
                        
                        return self.async_create_entry(title="", data={**self.config_entry.options, **sensor_data})
                    
            except InvalidSensor as err:
                errors["co2_sensors"] = "invalid_sensor"
                _LOGGER.warning(f"Invalid CO2 sensor: {err}")
            except Exception as err:
                errors["base"] = "unknown"
                _LOGGER.error(f"Unexpected error in CO2 sensors options: {err}", exc_info=True)

        # Show the CO2 sensor selection form
        return self.async_show_form(
            step_id="co2_sensors_options",
            data_schema=self._schema_co2_sensors_options(),
            errors=errors,
        )

    def _schema_co2_sensors_options(self) -> vol.Schema:
        """Generate schema for CO2 sensor selection with current values as defaults."""
        cur = self._data
        return vol.Schema({
            vol.Optional("co2_sensors", default=cur.get("co2_sensors", [])): selector.EntitySelector(
                selector.EntitySelectorConfig(
                    domain="sensor",
                    device_class="carbon_dioxide",
                    multiple=True,
                )
            ),
        }, extra=vol.ALLOW_EXTRA)

    async def async_step_co2_sensors_zones_options(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle CO2 sensor zone assignment in options flow."""
        errors: dict[str, str] = {}
        
        if user_input is not None:
            try:
                # Process zone assignments
                sensors = self._data.get("co2_sensors", [])
                zones = []
                for i in range(len(sensors)):
                    zone_key = f"co2_sensor_{i}_zone"
                    zone_value = user_input.get(zone_key, "1")
                    try:
                        zones.append(int(zone_value))
                    except ValueError:
                        zones.append(1)
                
                # Store zone assignments
                self._data["co2_sensor_zones"] = zones
                
                # Check if there are other newly enabled sensor types to configure
                old_data = {**self.config_entry.data, **self.config_entry.options}
                remaining_new_types = []
                for sensor_type in ("use_voc_sensors", "use_pm_sensors", "use_humidity_sensors"):
                    if (self._data.get(sensor_type) and not old_data.get(sensor_type, False)):
                        remaining_new_types.append(sensor_type)
                
                if remaining_new_types:
                    # Continue to next newly enabled sensor type
                    return await self._next_sensor_step_options("co2")
                else:
                    # No more new sensor types, save and return to menu
                    sensor_data = {}
                    for key in ("use_co2_sensors", "use_voc_sensors", "use_pm_sensors", "use_humidity_sensors",
                               "co2_sensors", "voc_sensors", "pm_sensors", "humidity_sensors",
                               "co2_sensor_zones", "voc_sensor_zones", "pm_sensor_zones", "humidity_sensor_zones"):
                        if key in self._data:
                            sensor_data[key] = self._data[key]
                    
                    return self.async_create_entry(title="", data={**self.config_entry.options, **sensor_data})
                    
            except Exception as err:
                errors["base"] = "unknown"
                _LOGGER.error(f"Unexpected error in CO2 zones options: {err}", exc_info=True)

        # Show the zone assignment form
        return self.async_show_form(
            step_id="co2_sensors_zones_options",
            data_schema=self._schema_co2_sensors_zones_options(),
            errors=errors,
        )

    def _schema_co2_sensors_zones_options(self) -> vol.Schema:
        """Generate schema for CO2 sensor zone assignment with current values as defaults."""
        cur = self._data
        sensors = cur.get("co2_sensors", [])
        current_zones = cur.get("co2_sensor_zones", [])
        num_zones = cur.get("num_zones", 1)
        
        if not sensors or num_zones <= 1:
            return vol.Schema({}, extra=vol.ALLOW_EXTRA)
        
        zone_options = get_zone_options(num_zones)
        schema_dict = {}
        
        for i, sensor in enumerate(sensors):
            current_zone = str(current_zones[i]) if i < len(current_zones) else "1"
            field_key = f"co2_sensor_{i}_zone"
            
            schema_dict[vol.Required(field_key, default=current_zone)] = selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=zone_options,
                    mode=selector.SelectSelectorMode.DROPDOWN,
                )
            )
        
        return vol.Schema(schema_dict, extra=vol.ALLOW_EXTRA)

    # VOC sensor options -------------------------------------------------------
    async def async_step_voc_sensors_options(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle VOC sensor selection in options flow."""
        errors: dict[str, str] = {}
        
        if user_input is not None:
            try:
                # Store VOC sensors
                sensors = user_input.get("voc_sensors", [])
                if not isinstance(sensors, list):
                    sensors = [sensors] if sensors else []
                sensors = [s for s in sensors if s]
                
                # Enable VOC sensors if any are selected
                if sensors:
                    self._data["use_voc_sensors"] = True
                    self._data["voc_sensors"] = sensors
                else:
                    # If no sensors selected, disable VOC sensors
                    self._data["use_voc_sensors"] = False
                    self._data["voc_sensors"] = []
                
                # Handle zone assignments if multiple zones and sensors exist
                num_zones = self._data.get("num_zones", 1)
                if num_zones > 1 and sensors:
                    return await self.async_step_voc_sensors_zones_options()
                else:
                    # Single zone or no sensors, assign all to zone 1
                    self._data["voc_sensor_zones"] = [1] * len(sensors)
                    
                    # Continue to next newly enabled sensor type or finish
                    return await self._next_sensor_step_options("voc")
                    
            except InvalidSensor as err:
                errors["voc_sensors"] = "invalid_sensor"
                _LOGGER.warning(f"Invalid VOC sensor: {err}")
            except Exception as err:
                errors["base"] = "unknown"
                _LOGGER.error(f"Unexpected error in VOC sensors options: {err}", exc_info=True)

        # Show the VOC sensor selection form
        return self.async_show_form(
            step_id="voc_sensors_options",
            data_schema=self._schema_voc_sensors_options(),
            errors=errors,
        )

    def _schema_voc_sensors_options(self) -> vol.Schema:
        """Generate schema for VOC sensor selection with current values as defaults."""
        cur = self._data
        return vol.Schema({
            vol.Optional("voc_sensors", default=cur.get("voc_sensors", [])): selector.EntitySelector(
                selector.EntitySelectorConfig(
                    domain="sensor",
                    device_class=["volatile_organic_compounds", "volatile_organic_compounds_parts"],
                    multiple=True,
                )
            ),
        }, extra=vol.ALLOW_EXTRA)

    async def async_step_voc_sensors_zones_options(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle VOC sensor zone assignment in options flow."""
        errors: dict[str, str] = {}
        
        if user_input is not None:
            try:
                # Process zone assignments
                sensors = self._data.get("voc_sensors", [])
                zones = []
                for i in range(len(sensors)):
                    zone_key = f"voc_sensor_{i}_zone"
                    zone_value = user_input.get(zone_key, "1")
                    try:
                        zones.append(int(zone_value))
                    except ValueError:
                        zones.append(1)
                
                # Store zone assignments
                self._data["voc_sensor_zones"] = zones
                
                # Continue to next newly enabled sensor type or finish
                return await self._next_sensor_step_options("voc")
                    
            except Exception as err:
                errors["base"] = "unknown"
                _LOGGER.error(f"Unexpected error in VOC zones options: {err}", exc_info=True)

        # Show the zone assignment form
        return self.async_show_form(
            step_id="voc_sensors_zones_options",
            data_schema=self._schema_voc_sensors_zones_options(),
            errors=errors,
        )

    def _schema_voc_sensors_zones_options(self) -> vol.Schema:
        """Generate schema for VOC sensor zone assignment with current values as defaults."""
        cur = self._data
        sensors = cur.get("voc_sensors", [])
        current_zones = cur.get("voc_sensor_zones", [])
        num_zones = cur.get("num_zones", 1)
        
        if not sensors or num_zones <= 1:
            return vol.Schema({}, extra=vol.ALLOW_EXTRA)
        
        zone_options = get_zone_options(num_zones)
        schema_dict = {}
        
        for i, sensor in enumerate(sensors):
            current_zone = str(current_zones[i]) if i < len(current_zones) else "1"
            field_key = f"voc_sensor_{i}_zone"
            
            schema_dict[vol.Required(field_key, default=current_zone)] = selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=zone_options,
                    mode=selector.SelectSelectorMode.DROPDOWN,
                )
            )
        
        return vol.Schema(schema_dict, extra=vol.ALLOW_EXTRA)

    # PM sensor options --------------------------------------------------------
    async def async_step_pm_sensors_options(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle PM sensor selection in options flow."""
        errors: dict[str, str] = {}
        
        if user_input is not None:
            try:
                # Store PM sensors
                sensors = user_input.get("pm_sensors", [])
                if not isinstance(sensors, list):
                    sensors = [sensors] if sensors else []
                sensors = [s for s in sensors if s]
                
                # Enable PM sensors if any are selected
                if sensors:
                    self._data["use_pm_sensors"] = True
                    self._data["pm_sensors"] = sensors
                else:
                    # If no sensors selected, disable PM sensors
                    self._data["use_pm_sensors"] = False
                    self._data["pm_sensors"] = []
                
                # Handle zone assignments if multiple zones and sensors exist
                num_zones = self._data.get("num_zones", 1)
                if num_zones > 1 and sensors:
                    return await self.async_step_pm_sensors_zones_options()
                else:
                    # Single zone or no sensors, assign all to zone 1
                    self._data["pm_sensor_zones"] = [1] * len(sensors)
                    
                    # Continue to next newly enabled sensor type or finish
                    return await self._next_sensor_step_options("pm")
                    
            except InvalidSensor as err:
                errors["pm_sensors"] = "invalid_sensor"
                _LOGGER.warning(f"Invalid PM sensor: {err}")
            except Exception as err:
                errors["base"] = "unknown"
                _LOGGER.error(f"Unexpected error in PM sensors options: {err}", exc_info=True)

        # Show the PM sensor selection form
        return self.async_show_form(
            step_id="pm_sensors_options",
            data_schema=self._schema_pm_sensors_options(),
            errors=errors,
        )

    def _schema_pm_sensors_options(self) -> vol.Schema:
        """Generate schema for PM sensor selection with current values as defaults."""
        cur = self._data
        return vol.Schema({
            vol.Optional("pm_sensors", default=cur.get("pm_sensors", [])): selector.EntitySelector(
                selector.EntitySelectorConfig(
                    domain="sensor",
                    device_class=["pm1", "pm10", "pm25"],
                    multiple=True,
                )
            ),
        }, extra=vol.ALLOW_EXTRA)

    async def async_step_pm_sensors_zones_options(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle PM sensor zone assignment in options flow."""
        errors: dict[str, str] = {}
        
        if user_input is not None:
            try:
                # Process zone assignments
                sensors = self._data.get("pm_sensors", [])
                zones = []
                for i in range(len(sensors)):
                    zone_key = f"pm_sensor_{i}_zone"
                    zone_value = user_input.get(zone_key, "1")
                    try:
                        zones.append(int(zone_value))
                    except ValueError:
                        zones.append(1)
                
                # Store zone assignments
                self._data["pm_sensor_zones"] = zones
                
                # Continue to next newly enabled sensor type or finish
                return await self._next_sensor_step_options("pm")
                    
            except Exception as err:
                errors["base"] = "unknown"
                _LOGGER.error(f"Unexpected error in PM zones options: {err}", exc_info=True)

        # Show the zone assignment form
        return self.async_show_form(
            step_id="pm_sensors_zones_options",
            data_schema=self._schema_pm_sensors_zones_options(),
            errors=errors,
        )

    def _schema_pm_sensors_zones_options(self) -> vol.Schema:
        """Generate schema for PM sensor zone assignment with current values as defaults."""
        cur = self._data
        sensors = cur.get("pm_sensors", [])
        current_zones = cur.get("pm_sensor_zones", [])
        num_zones = cur.get("num_zones", 1)
        
        if not sensors or num_zones <= 1:
            return vol.Schema({}, extra=vol.ALLOW_EXTRA)
        
        zone_options = get_zone_options(num_zones)
        schema_dict = {}
        
        for i, sensor in enumerate(sensors):
            current_zone = str(current_zones[i]) if i < len(current_zones) else "1"
            field_key = f"pm_sensor_{i}_zone"
            
            schema_dict[vol.Required(field_key, default=current_zone)] = selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=zone_options,
                    mode=selector.SelectSelectorMode.DROPDOWN,
                )
            )
        
        return vol.Schema(schema_dict, extra=vol.ALLOW_EXTRA)

    # Humidity sensor options --------------------------------------------------
    async def async_step_humidity_sensors_options(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle Humidity sensor selection in options flow."""
        errors: dict[str, str] = {}
        
        if user_input is not None:
            try:
                # Store Humidity sensors
                sensors = user_input.get("humidity_sensors", [])
                if not isinstance(sensors, list):
                    sensors = [sensors] if sensors else []
                sensors = [s for s in sensors if s]
                
                # Enable Humidity sensors if any are selected
                if sensors:
                    self._data["use_humidity_sensors"] = True
                    self._data["humidity_sensors"] = sensors
                else:
                    # If no sensors selected, disable Humidity sensors
                    self._data["use_humidity_sensors"] = False
                    self._data["humidity_sensors"] = []
                
                # Handle zone assignments if multiple zones and sensors exist
                num_zones = self._data.get("num_zones", 1)
                if num_zones > 1 and sensors:
                    return await self.async_step_humidity_sensors_zones_options()
                else:
                    # Single zone or no sensors, assign all to zone 1
                    self._data["humidity_sensor_zones"] = [1] * len(sensors)
                    
                    # Humidity is the last sensor type, always save and complete
                    return await self._next_sensor_step_options("humidity")
                    
            except InvalidSensor as err:
                errors["humidity_sensors"] = "invalid_sensor"
                _LOGGER.warning(f"Invalid Humidity sensor: {err}")
            except Exception as err:
                errors["base"] = "unknown"
                _LOGGER.error(f"Unexpected error in Humidity sensors options: {err}", exc_info=True)

        # Show the Humidity sensor selection form
        return self.async_show_form(
            step_id="humidity_sensors_options",
            data_schema=self._schema_humidity_sensors_options(),
            errors=errors,
        )

    def _schema_humidity_sensors_options(self) -> vol.Schema:
        """Generate schema for Humidity sensor selection with current values as defaults."""
        cur = self._data
        return vol.Schema({
            vol.Optional("humidity_sensors", default=cur.get("humidity_sensors", [])): selector.EntitySelector(
                selector.EntitySelectorConfig(
                    domain="sensor",
                    device_class="humidity",
                    multiple=True,
                )
            ),
        }, extra=vol.ALLOW_EXTRA)

    async def async_step_humidity_sensors_zones_options(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle Humidity sensor zone assignment in options flow."""
        errors: dict[str, str] = {}
        
        if user_input is not None:
            try:
                # Process zone assignments
                sensors = self._data.get("humidity_sensors", [])
                zones = []
                for i in range(len(sensors)):
                    zone_key = f"humidity_sensor_{i}_zone"
                    zone_value = user_input.get(zone_key, "1")
                    try:
                        zones.append(int(zone_value))
                    except ValueError:
                        zones.append(1)
                
                # Store zone assignments
                self._data["humidity_sensor_zones"] = zones
                
                # Humidity is the last sensor type, always save and complete
                return await self._next_sensor_step_options("humidity")
                    
            except Exception as err:
                errors["base"] = "unknown"
                _LOGGER.error(f"Unexpected error in Humidity zones options: {err}", exc_info=True)

        # Show the zone assignment form
        return self.async_show_form(
            step_id="humidity_sensors_zones_options",
            data_schema=self._schema_humidity_sensors_zones_options(),
            errors=errors,
        )

    def _schema_humidity_sensors_zones_options(self) -> vol.Schema:
        """Generate schema for Humidity sensor zone assignment with current values as defaults."""
        cur = self._data
        sensors = cur.get("humidity_sensors", [])
        current_zones = cur.get("humidity_sensor_zones", [])
        num_zones = cur.get("num_zones", 1)
        
        if not sensors or num_zones <= 1:
            return vol.Schema({}, extra=vol.ALLOW_EXTRA)
        
        zone_options = get_zone_options(num_zones)
        schema_dict = {}
        
        for i, sensor in enumerate(sensors):
            current_zone = str(current_zones[i]) if i < len(current_zones) else "1"
            field_key = f"humidity_sensor_{i}_zone"
            
            schema_dict[vol.Required(field_key, default=current_zone)] = selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=zone_options,
                    mode=selector.SelectSelectorMode.DROPDOWN,
                )
            )
        
        return vol.Schema(schema_dict, extra=vol.ALLOW_EXTRA)

    # Helper method for navigation ----------------------------------------------
    async def _next_sensor_step_options(self, completed_sensor_type: str) -> FlowResult:
        """Navigate to the next newly enabled sensor type or finish sensor configuration."""
        # Get current vs old configuration to determine newly enabled types
        old_data = {**self.config_entry.data, **self.config_entry.options}
        
        # Find remaining newly enabled sensor types after the completed one
        sensor_order = ["co2", "voc", "pm", "humidity"]
        completed_index = sensor_order.index(completed_sensor_type) if completed_sensor_type in sensor_order else -1
        
        for i, sensor_name in enumerate(sensor_order):
            if i <= completed_index:
                continue  # Skip completed and earlier sensor types
                
            sensor_type_key = f"use_{sensor_name}_sensors"
            # Only go to this sensor type if it's newly enabled (enabled now but wasn't before)
            if (self._data.get(sensor_type_key) and not old_data.get(sensor_type_key, False)):
                if sensor_name == "voc":
                    return await self.async_step_voc_sensors_options()
                elif sensor_name == "pm":
                    return await self.async_step_pm_sensors_options()
                elif sensor_name == "humidity":
                    return await self.async_step_humidity_sensors_options()
        
        # No more newly enabled sensor types, save and complete
        sensor_data = {}
        for key in ("use_co2_sensors", "use_voc_sensors", "use_pm_sensors", "use_humidity_sensors",
                   "co2_sensors", "voc_sensors", "pm_sensors", "humidity_sensors",
                   "co2_sensor_zones", "voc_sensor_zones", "pm_sensor_zones", "humidity_sensor_zones"):
            if key in self._data:
                sensor_data[key] = self._data[key]
        
        return self.async_create_entry(title="", data={**self.config_entry.options, **sensor_data})

    # Device configuration options --------------------------------------------------
    async def async_step_device_configuration(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle device configuration in options flow."""
        errors: dict[str, str] = {}
        
        if user_input is not None:
            try:
                # Validate the device selection input
                info = await validate_device_selection_input(self.hass, user_input)
                
                # Store device selection data
                updated_data = {}
                updated_data["remote_device"] = user_input["remote_device"]
                
                # Handle num_zones changes
                new_num_zones = int(user_input.get("num_zones", 1))
                old_num_zones = self._data.get("num_zones", 1)
                updated_data["num_zones"] = new_num_zones
                
                # Extract remote device serial number for prefilling zone configs
                remote_serial = await get_device_serial_number(self.hass, user_input["remote_device"])
                updated_data["remote_serial"] = remote_serial or ""
                
                # Handle zone_configs changes when num_zones changes
                current_zone_configs = self._data.get("zone_configs", {})
                
                # Always clean up zone configs to match the target number
                # This prevents sensor duplication bugs
                new_zone_configs = {}
                
                # Copy/update existing zone configs up to the new number of zones only
                for zone_num in range(1, new_num_zones + 1):
                    if zone_num in current_zone_configs:
                        new_zone_configs[zone_num] = current_zone_configs[zone_num].copy()
                        # Update remote device serial in existing configs
                        new_zone_configs[zone_num]["id_from"] = remote_serial or new_zone_configs[zone_num].get("id_from", "")
                    else:
                        # Create new zone config for additional zones
                        new_zone_configs[zone_num] = {
                            "id_from": remote_serial or "",
                            "id_to": "",  # Must be configured manually in zone config
                            "sensor_id": 256 - zone_num,  # Default sensor IDs: 255, 254, 253, etc.
                            "min_fan_rate": 0,
                            "max_fan_rate": 100,  # Default to 100% instead of 255
                        }
                
                # Log zone changes for debugging
                removed_zones = set(current_zone_configs.keys()) - set(new_zone_configs.keys())
                added_zones = set(new_zone_configs.keys()) - set(current_zone_configs.keys())
                if removed_zones:
                    _LOGGER.info(f"Device config: removed zone configs for {sorted(removed_zones)}")
                if added_zones:
                    _LOGGER.info(f"Device config: added zone configs for {sorted(added_zones)}")
                
                updated_data["zone_configs"] = new_zone_configs
                
                # Validate the new configuration before creating entry (allow cleanup in options flow)
                new_options = {**self.config_entry.options, **updated_data}
                temp_cfg = merge_config_with_validation(self.config_entry.data, new_options, allow_cleanup=True)
                zone_configs = temp_cfg.get("zone_configs", {})
                temp_zones = len(zone_configs) if zone_configs else int(temp_cfg.get("num_zones", 1))
                
                # Ensure zone configs match expected count
                if len(zone_configs) != temp_zones:
                    _LOGGER.error(f"Zone validation failed: configs={len(zone_configs)}, expected={temp_zones}")
                    errors["base"] = "Zone configuration validation failed"
                else:
                    # Create the entry with validated data
                    _LOGGER.info(f"Device configuration updated: {temp_zones} zones validated")
                    return self.async_create_entry(title="", data=new_options)
                
            except InvalidEntity as err:
                # Determine which device field had the error
                error_msg = str(err).lower()
                if "remote" in error_msg:
                    errors["remote_device"] = str(err)
                else:
                    errors["base"] = str(err)
                _LOGGER.warning(f"Invalid entity: {err}")
            except Exception as err:
                errors["base"] = "unknown"
                _LOGGER.error(f"Unexpected error in device selection options: {err}", exc_info=True)

        # Show the device configuration form with current values as defaults
        return self.async_show_form(
            step_id="device_configuration",
            data_schema=self._schema_device_selection(),
            errors=errors,
        )

    def _schema_device_selection(self) -> vol.Schema:
        """Generate schema for device selection options with current values as defaults."""
        cur = self._data
        return vol.Schema({
            vol.Required("remote_device", default=cur.get("remote_device")): selector.DeviceSelector(
                selector.DeviceSelectorConfig(integration="ramses_cc")
            ),
            vol.Optional("num_zones", default=str(cur.get("num_zones", 1))): selector.SelectSelector(
                selector.SelectSelectorConfig(
                    options=[
                        selector.SelectOptionDict(value="1", label="1 Zone"),
                        selector.SelectOptionDict(value="2", label="2 Zones"), 
                        selector.SelectOptionDict(value="3", label="3 Zones"),
                        selector.SelectOptionDict(value="4", label="4 Zones"),
                        selector.SelectOptionDict(value="5", label="5 Zones"),
                    ],
                    mode=selector.SelectSelectorMode.DROPDOWN
                )
            ),
        }, extra=vol.ALLOW_EXTRA)

    # Zone configuration options ----------------------------------------------
    async def async_step_zone_1_config(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle zone 1 configuration in options flow."""
        return await self._async_step_zone_config_options(1, user_input)

    async def async_step_zone_2_config(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle zone 2 configuration in options flow."""
        return await self._async_step_zone_config_options(2, user_input)

    async def async_step_zone_3_config(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle zone 3 configuration in options flow."""
        return await self._async_step_zone_config_options(3, user_input)

    async def async_step_zone_4_config(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle zone 4 configuration in options flow."""
        return await self._async_step_zone_config_options(4, user_input)

    async def async_step_zone_5_config(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Handle zone 5 configuration in options flow."""
        return await self._async_step_zone_config_options(5, user_input)

    async def _async_step_zone_config_options(
        self, zone_number: int, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle zone configuration step for any zone number in options flow."""
        errors: dict[str, str] = {}
        
        # Validate zone number against current configuration to prevent bugs
        if zone_number > self._validated_num_zones:
            _LOGGER.error(f"Attempted to configure zone {zone_number} but only {self._validated_num_zones} zones are available")
            return self.async_abort(reason="invalid_zone")
        
        if user_input is not None:
            try:
                # Validate the zone configuration input
                info = await validate_zone_config_input(self.hass, user_input, zone_number)
                
                # Extract device serials for storage
                device_from_id = user_input["device_from"]
                device_to_id = user_input["device_to"]
                id_from = await get_device_serial_number(self.hass, device_from_id)
                id_to = await get_device_serial_number(self.hass, device_to_id)
                
                # Update zone configuration data
                current_zone_configs = self._data.get("zone_configs", {})
                current_zone_configs[zone_number] = {
                    "device_from": device_from_id,  # Store device ID for config flow
                    "device_to": device_to_id,      # Store device ID for config flow
                    "id_from": id_from,             # Store serial for commands
                    "id_to": id_to,                 # Store serial for commands
                    "sensor_id": user_input["sensor_id"],
                    "min_fan_rate": user_input["min_fan_rate"],
                    "max_fan_rate": user_input["max_fan_rate"],
                }
                
                # Preserve all existing options and only update zone_configs
                updated_options = dict(self.config_entry.options)
                updated_options["zone_configs"] = current_zone_configs
                
                # Validate zone configuration consistency before saving (no cleanup for individual zone updates)
                temp_cfg = merge_config_with_validation(self.config_entry.data, updated_options, allow_cleanup=False)
                zone_configs = temp_cfg.get("zone_configs", {})
                temp_zones = len(zone_configs) if zone_configs else int(temp_cfg.get("num_zones", 1))
                
                if len(zone_configs) != temp_zones:
                    _LOGGER.error(f"Zone {zone_number} config validation failed")
                    errors["base"] = "Zone configuration validation failed"
                else:
                    # Create the entry with properly preserved options
                    _LOGGER.info(f"Zone {zone_number} configuration updated successfully")
                    return self.async_create_entry(title="", data=updated_options)
                
            except InvalidZoneConfig as err:
                # Provide specific error messages based on validation
                error_msg = str(err).lower()
                if "'from' device" in error_msg:
                    errors["device_from"] = str(err)
                elif "'to' device" in error_msg:
                    errors["device_to"] = str(err)
                elif "sensor_id" in error_msg:
                    errors["sensor_id"] = str(err)
                elif "min_fan" in error_msg:
                    errors["min_fan_rate"] = str(err)
                elif "max_fan" in error_msg:
                    errors["max_fan_rate"] = str(err)
                else:
                    errors["base"] = str(err)
                _LOGGER.warning(f"Invalid zone config: {err}")
            except Exception as err:
                errors["base"] = "unknown"
                _LOGGER.error(f"Unexpected error in zone {zone_number} config options: {err}", exc_info=True)

        # Show the zone configuration form with current values as defaults
        return self.async_show_form(
            step_id=f"zone_{zone_number}_config",
            data_schema=self._schema_zone_config(zone_number),
            errors=errors,
        )

    def _schema_zone_config(self, zone_number: int) -> vol.Schema:
        """Generate schema for zone configuration options with current values as defaults."""
        cur = self._data
        current_zone_config = cur.get("zone_configs", {}).get(zone_number, {})
        remote_device_id = cur.get("remote_device")
        
        # Use device IDs if available, otherwise fall back to remote device for device_from
        default_device_from = current_zone_config.get("device_from", remote_device_id)
        default_device_to = current_zone_config.get("device_to", None)
        
        # Calculate default sensor ID: 255 for zone 1, 254 for zone 2, etc.
        default_sensor_id = current_zone_config.get("sensor_id", 256 - zone_number)
        
        # Convert stored percentage values (0-100) to display defaults
        default_min_fan_rate = current_zone_config.get("min_fan_rate", 0)
        default_max_fan_rate = current_zone_config.get("max_fan_rate", 100)
        
        return vol.Schema({
            vol.Required("device_from", default=default_device_from): selector.DeviceSelector(
                selector.DeviceSelectorConfig(integration="ramses_cc")
            ),
            vol.Required("device_to", default=default_device_to): selector.DeviceSelector(
                selector.DeviceSelectorConfig(integration="ramses_cc")
            ),
            vol.Required("sensor_id", default=default_sensor_id): vol.Coerce(int),
            vol.Required("min_fan_rate", default=default_min_fan_rate): vol.All(vol.Coerce(int), vol.Range(min=0, max=100)),
            vol.Required("max_fan_rate", default=default_max_fan_rate): vol.All(vol.Coerce(int), vol.Range(min=0, max=100)),
        }, extra=vol.ALLOW_EXTRA)