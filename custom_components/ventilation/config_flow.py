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

from .const import (
    DOMAIN,
    DEFAULT_CO2_INDEX,
    DEFAULT_VOC_PPM_INDEX,
    DEFAULT_VOC_INDEX,
    DEFAULT_PM_1_0_INDEX,
    DEFAULT_PM_2_5_INDEX,
    DEFAULT_PM_10_INDEX,
)
_LOGGER = logging.getLogger(__name__)

def get_basic_schema():
    """Generate schema for fan settings."""
    return vol.Schema({
        vol.Required("min_fan_output", default=0): vol.Coerce(int),
        vol.Required("max_fan_output", default=255): vol.Coerce(int),
        vol.Required("remote_device"): selector.DeviceSelector(
            selector.DeviceSelectorConfig(integration="ramses_cc")
        ),
        vol.Required("fan_device"): selector.DeviceSelector(
            selector.DeviceSelectorConfig(integration="ramses_cc")
        ),
    }, extra=vol.ALLOW_EXTRA)


def get_pid_parameters_schema():
    """Generate schema for PID parameters (including Ki times)."""
    return vol.Schema({
        vol.Required("setpoint", default=0.5): vol.Coerce(float),
        vol.Required("kp", default=25.5): vol.Coerce(float),
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
        vol.Optional("back", default=False): selector.BooleanSelector(),
    }, extra=vol.ALLOW_EXTRA)


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
    
    # Add back button
    schema_dict[vol.Optional("back", default=False)] = selector.BooleanSelector()
    
    return vol.Schema(schema_dict, extra=vol.ALLOW_EXTRA)


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
    if not any([data.get("use_co2_sensors"), data.get("use_voc_sensors"), data.get("use_pm_sensors")]):
        raise InvalidSensor("At least one sensor type must be selected")
    
    return {"title": "PID Ventilation Control"}


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
            
            # Validate that values are in ascending order (except for the last one which can be higher)
            if i > 0 and index_value <= index_values[-1]:
                raise InvalidAirQualityIndex(f"{index_name} values must be in ascending order")
            
            # Validate range based on index type
            if index_name.startswith("CO2") and not (0 <= index_value <= 10000):
                raise InvalidAirQualityIndex(f"CO2 values must be between 0 and 10000 ppm")
            elif index_name.startswith("VOC") and not (0 <= index_value <= 1000):
                raise InvalidAirQualityIndex(f"VOC values must be between 0 and 1000 ppm")
            elif index_name.startswith("PM") and not (0 <= index_value <= 1000):
                raise InvalidAirQualityIndex(f"PM values must be between 0 and 1000 µg/m³")
            
            index_values.append(index_value)
        
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


class ConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for PID Ventilation Control."""

    VERSION = 1

    def __init__(self):
        """Initialize the config flow."""
        self._data = {}
        self._sensor_types = {}
        # Track which specific sensor classes were selected
        self._sensor_classes = {
            "pm1": False,
            "pm25": False,
            "pm10": False,
            "voc": False,
            "voc_parts": False,
        }

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        """Get the options flow for this handler."""
        return OptionsFlowHandler(config_entry)

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step - fan settings."""
        errors: dict[str, str] = {}
        
        # Check if already configured (only allow one instance)
        if self._async_current_entries():
            return self.async_abort(reason="already_configured")
        
        if user_input is not None:
            try:
                # Validate the fan settings input
                info = await validate_basic_input(self.hass, user_input)
                
                # Store fan settings data
                self._data.update(user_input)
                
                # Go to sensor type selection step
                return await self.async_step_sensor_types()
            except InvalidEntity as err:
                errors["entity_id"] = "invalid_entity"
                _LOGGER.warning(f"Invalid entity: {err}")
            except InvalidRange as err:
                errors["min_fan_output"] = "invalid_range"
                _LOGGER.warning(f"Invalid range: {err}")
            except Exception as err:  # pylint: disable=broad-except
                errors["base"] = "unknown"
                _LOGGER.error(f"Unexpected error in config flow: {err}", exc_info=True)

        # Show the fan settings form
        return self.async_show_form(
            step_id="user",
            data_schema=get_basic_schema(),
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

                sensor_types_str = "/".join(sensor_types) if sensor_types else "sensors"
                title = f"PID Ventilation Control ({total_sensors} {sensor_types_str} sensors)"

                return self.async_create_entry(title=title, data=self._data)
            except InvalidRange as err:
                # Determine which field has the error based on the error message
                error_msg = str(err).lower()
                if "setpoint" in error_msg:
                    errors["setpoint"] = "invalid_range"
                elif "kp" in error_msg:
                    errors["kp"] = "invalid_range"
                elif "interval" in error_msg:
                    errors["update_interval"] = "invalid_range"
                else:
                    errors["base"] = "invalid_range"
                _LOGGER.warning(f"Invalid range: {err}")
            except InvalidKiTimes as err:
                errors["ki_times"] = "invalid_ki_times"
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
                return await self.async_step_user()
                
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
                self._data["co2_sensors"] = user_input["co2_sensors"]
                
                # Go to next sensor type or air quality indices step
                if self._sensor_types.get("use_voc_sensors"):
                    return await self.async_step_voc_sensors()
                elif self._sensor_types.get("use_pm_sensors"):
                    return await self.async_step_pm_sensors()
                else:
                    return await self.async_step_air_quality_indices()
                    
            except InvalidSensor as err:
                errors["co2_sensors"] = "invalid_sensor"
                _LOGGER.warning(f"Invalid CO2 sensor: {err}")
            except Exception as err:  # pylint: disable=broad-except
                errors["base"] = "unknown"
                _LOGGER.error(f"Unexpected error in CO2 sensor step: {err}", exc_info=True)

        # Show the CO2 sensor selection form
        return self.async_show_form(
            step_id="co2_sensors",
            data_schema=get_co2_sensors_schema(self.hass),
            errors=errors,
        )

    async def async_step_voc_sensors(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the VOC sensor selection step."""
        errors: dict[str, str] = {}
        
        if user_input is not None:
            # Check for back button
            if user_input.get("back"):
                # Go back to CO2 step if it was selected, otherwise to sensor_types
                if self._sensor_types.get("use_co2_sensors"):
                    return await self.async_step_co2_sensors()
                else:
                    return await self.async_step_sensor_types()
                
            try:
                # Validate the sensor input
                info = await validate_sensors_input(self.hass, user_input, "voc")
                
                # Store VOC sensors
                self._data["voc_sensors"] = user_input["voc_sensors"]
                
                # Detect VOC device classes selected
                self._update_voc_classes(user_input["voc_sensors"])
                
                # Go to next sensor type or air quality indices step
                if self._sensor_types.get("use_pm_sensors"):
                    return await self.async_step_pm_sensors()
                else:
                    return await self.async_step_air_quality_indices()
                    
            except InvalidSensor as err:
                errors["voc_sensors"] = "invalid_sensor"
                _LOGGER.warning(f"Invalid VOC sensor: {err}")
            except Exception as err:  # pylint: disable=broad-except
                errors["base"] = "unknown"
                _LOGGER.error(f"Unexpected error in VOC sensor step: {err}", exc_info=True)

        # Show the VOC sensor selection form
        return self.async_show_form(
            step_id="voc_sensors",
            data_schema=get_voc_sensors_schema(self.hass),
            errors=errors,
        )

    async def async_step_pm_sensors(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the PM sensor selection step."""
        errors: dict[str, str] = {}
        
        if user_input is not None:
            # Check for back button
            if user_input.get("back"):
                # Go back to VOC step if it was selected, else CO2 step if selected, else sensor_types
                if self._sensor_types.get("use_voc_sensors"):
                    return await self.async_step_voc_sensors()
                elif self._sensor_types.get("use_co2_sensors"):
                    return await self.async_step_co2_sensors()
                else:
                    return await self.async_step_sensor_types()
                
            try:
                # Validate the sensor input
                info = await validate_sensors_input(self.hass, user_input, "pm")
                
                # Store PM sensors
                self._data["pm_sensors"] = user_input["pm_sensors"]
                
                # Detect PM device classes selected
                self._update_pm_classes(user_input["pm_sensors"])
                
                # Go to air quality indices step
                return await self.async_step_air_quality_indices()
                    
            except InvalidSensor as err:
                errors["pm_sensors"] = "invalid_sensor"
                _LOGGER.warning(f"Invalid PM sensor: {err}")
            except Exception as err:  # pylint: disable=broad-except
                errors["base"] = "unknown"
                _LOGGER.error(f"Unexpected error in PM sensor step: {err}", exc_info=True)

        # Show the PM sensor selection form
        return self.async_show_form(
            step_id="pm_sensors",
            data_schema=get_pm_sensors_schema(self.hass),
            errors=errors,
        )

    async def async_step_air_quality_indices(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the air quality indices configuration step."""
        errors: dict[str, str] = {}
        
        if user_input is not None:
            # Check for back button
            if user_input.get("back"):
                # Go back to the last sensor type step that was configured
                if self._sensor_types.get("use_pm_sensors"):
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
            except InvalidAirQualityIndex:
                errors["co2_index"] = "invalid_air_quality_index"
                valid = False
            
            try:
                if self._sensor_types.get("use_voc_sensors") and self._sensor_classes["voc"]:
                    voc_index = validate_air_quality_index(user_input["voc_index"], "VOC")
                    self._data["voc_index"] = voc_index
            except InvalidAirQualityIndex:
                errors["voc_index"] = "invalid_air_quality_index"
                valid = False
            
            try:
                if self._sensor_types.get("use_voc_sensors") and self._sensor_classes["voc_parts"]:
                    voc_ppm_index = validate_air_quality_index(user_input["voc_ppm_index"], "VOC (parts)")
                    self._data["voc_ppm_index"] = voc_ppm_index
            except InvalidAirQualityIndex:
                errors["voc_ppm_index"] = "invalid_air_quality_index"
                valid = False
            
            try:
                if self._sensor_types.get("use_pm_sensors") and self._sensor_classes["pm1"]:
                    pm_1_0_index = validate_air_quality_index(user_input["pm_1_0_index"], "PM1.0")
                    self._data["pm_1_0_index"] = pm_1_0_index
            except InvalidAirQualityIndex:
                errors["pm_1_0_index"] = "invalid_air_quality_index"
                valid = False
            
            try:
                if self._sensor_types.get("use_pm_sensors") and self._sensor_classes["pm25"]:
                    pm_2_5_index = validate_air_quality_index(user_input["pm_2_5_index"], "PM2.5")
                    self._data["pm_2_5_index"] = pm_2_5_index
            except InvalidAirQualityIndex:
                errors["pm_2_5_index"] = "invalid_air_quality_index"
                valid = False
            
            try:
                if self._sensor_types.get("use_pm_sensors") and self._sensor_classes["pm10"]:
                    pm_10_index = validate_air_quality_index(user_input["pm_10_index"], "PM10", expected_count=5)
                    self._data["pm_10_index"] = pm_10_index
            except InvalidAirQualityIndex:
                errors["pm_10_index"] = "invalid_air_quality_index"
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

    def _update_voc_classes(self, entity_ids: list[str]) -> None:
        classes = self._detect_device_classes(entity_ids)
        self._sensor_classes["voc"] = "volatile_organic_compounds" in classes
        self._sensor_classes["voc_parts"] = "volatile_organic_compounds_parts" in classes

    def _update_pm_classes(self, entity_ids: list[str]) -> None:
        classes = self._detect_device_classes(entity_ids)
        self._sensor_classes["pm1"] = "pm1" in classes
        self._sensor_classes["pm25"] = "pm25" in classes
        self._sensor_classes["pm10"] = "pm10" in classes

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


class OptionsFlowHandler(config_entries.OptionsFlow):
    """Handle options flow for PID Ventilation Control."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry
        # Start from current saved config merged with options
        self._data: dict[str, Any] = {**config_entry.data, **config_entry.options}

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        """Top-level options menu."""
        return self.async_show_menu(
            step_id="init",
            menu_options=[
                "fan_settings",
                "sensor_settings",
                "iaq_settings",
                "pid_settings",
            ],
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
            vol.Optional("fan_device", default=cur.get("fan_device")): selector.DeviceSelector(
                selector.DeviceSelectorConfig(integration="ramses_cc")
            ),
        }, extra=vol.ALLOW_EXTRA)

    async def async_step_fan_settings(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        errors: dict[str, str] = {}
        if user_input is not None:
            try:
                await validate_basic_input(self.hass, user_input)
                # Create options dict with only the changed settings
                options = {**self.config_entry.options}
                options.update(user_input)
                return self.async_create_entry(title="", data=options)
            except InvalidEntity:
                errors["remote_device"] = "invalid_entity"
            except InvalidRange:
                errors["min_fan_output"] = "invalid_range"
            except Exception as e:
                _LOGGER.error(f"Unknown error in fan_settings: {e}", exc_info=True)
                errors["base"] = "unknown"
        return self.async_show_form(step_id="fan_settings", data_schema=self._schema_fan(), errors=errors)

    # Sensor settings --------------------------------------------------------
    def _schema_sensors(self) -> vol.Schema:
        cur = self._data
        return vol.Schema({
            vol.Optional("use_co2_sensors", default=cur.get("use_co2_sensors", bool(cur.get("co2_sensors")))): cv.boolean,
            vol.Optional("use_voc_sensors", default=cur.get("use_voc_sensors", bool(cur.get("voc_sensors")))): cv.boolean,
            vol.Optional("use_pm_sensors", default=cur.get("use_pm_sensors", bool(cur.get("pm_sensors")))): cv.boolean,
            vol.Optional("co2_sensors", default=cur.get("co2_sensors", [])): selector.EntitySelector(
                selector.EntitySelectorConfig(domain="sensor", device_class="carbon_dioxide", multiple=True)
            ),
            vol.Optional("voc_sensors", default=cur.get("voc_sensors", [])): selector.EntitySelector(
                selector.EntitySelectorConfig(domain="sensor", device_class=["volatile_organic_compounds", "volatile_organic_compounds_parts"], multiple=True)
            ),
            vol.Optional("pm_sensors", default=cur.get("pm_sensors", [])): selector.EntitySelector(
                selector.EntitySelectorConfig(domain="sensor", device_class=["pm1", "pm10", "pm25"], multiple=True)
            ),
        }, extra=vol.ALLOW_EXTRA)

    async def async_step_sensor_settings(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        errors: dict[str, str] = {}
        if user_input is not None:
            try:
                # Persist booleans
                for k in ("use_co2_sensors", "use_voc_sensors", "use_pm_sensors"):
                    if k in user_input:
                        self._data[k] = user_input[k]
                # Persist sensors (lists)
                for k in ("co2_sensors", "voc_sensors", "pm_sensors"):
                    if k in user_input:
                        sensors = user_input.get(k) or []
                        if not isinstance(sensors, list):
                            sensors = [sensors]
                        self._data[k] = [s for s in sensors if s]
                # Basic validation: if a type is enabled, require at least one
                if self._data.get("use_co2_sensors") and not self._data.get("co2_sensors"):
                    raise InvalidSensor("At least one CO2 sensor must be specified")
                if self._data.get("use_voc_sensors") and not self._data.get("voc_sensors"):
                    raise InvalidSensor("At least one VOC sensor must be specified")
                if self._data.get("use_pm_sensors") and not self._data.get("pm_sensors"):
                    raise InvalidSensor("At least one PM sensor must be specified")
                return self.async_create_entry(title="", data={**self.config_entry.options, **{k: v for k, v in self._data.items() if k in user_input or k.endswith('_sensors')}})
            except InvalidSensor:
                errors["base"] = "no_sensors"
            except Exception as e:
                _LOGGER.error(f"Unknown error in sensor_settings: {e}", exc_info=True)
                errors["base"] = "unknown"
        return self.async_show_form(step_id="sensor_settings", data_schema=self._schema_sensors(), errors=errors)

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
        }, extra=vol.ALLOW_EXTRA)

    async def async_step_iaq_settings(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        errors: dict[str, str] = {}
        if user_input is not None:
            try:
                # Only validate and update provided fields
                for key, name in (
                    ("co2_index", "CO2"),
                    ("voc_index", "VOC"),
                    ("voc_ppm_index", "VOC PPM"),
                    ("pm_1_0_index", "PM1.0"),
                    ("pm_2_5_index", "PM2.5"),
                    ("pm_10_index", "PM10"),
                ):
                    if key in user_input:
                        self._data[key] = validate_air_quality_index(user_input[key], name)
                return self.async_create_entry(title="", data={**self.config_entry.options, **{key: self._data[key] for key in self._data if key.endswith('_index')}})
            except InvalidAirQualityIndex:
                errors["base"] = "invalid_air_quality_index"
            except Exception as e:
                _LOGGER.error(f"Unknown error in iaq_settings: {e}", exc_info=True)
                errors["base"] = "unknown"
        return self.async_show_form(step_id="iaq_settings", data_schema=self._schema_iaq(), errors=errors)

    # PID settings -----------------------------------------------------------
    def _schema_pid(self) -> vol.Schema:
        cur = self._data
        return vol.Schema({
            vol.Required("setpoint", default=cur.get("setpoint", 0.5)): vol.Coerce(float),
            vol.Required("kp", default=cur.get("kp", 25.5)): vol.Coerce(float),
            vol.Required("ki_times", default=",".join(map(str, cur.get("ki_times", [3600, 1800, 900, 450, 225, 150])))): cv.string,
            vol.Required("update_interval", default=cur.get("update_interval", 300)): vol.Coerce(int),
        }, extra=vol.ALLOW_EXTRA)

    async def async_step_pid_settings(self, user_input: dict[str, Any] | None = None) -> FlowResult:
        errors: dict[str, str] = {}
        if user_input is not None:
            try:
                await validate_pid_parameters_input(self.hass, user_input)
                # Process ki_times string into list
                validated_data = {
                    "setpoint": user_input["setpoint"],
                    "kp": user_input["kp"],
                    "update_interval": user_input["update_interval"],
                    "ki_times": validate_ki_times(user_input["ki_times"]),
                }
                return self.async_create_entry(title="", data={**self.config_entry.options, **validated_data})
            except InvalidRange:
                errors["base"] = "invalid_range"
            except InvalidKiTimes:
                errors["ki_times"] = "invalid_ki_times"
            except Exception as e:
                _LOGGER.error(f"Unknown error in pid_settings: {e}", exc_info=True)
                errors["base"] = "unknown"
        return self.async_show_form(step_id="pid_settings", data_schema=self._schema_pid(), errors=errors)