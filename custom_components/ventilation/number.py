"""Number platform for PID Ventilation Control integration."""
from __future__ import annotations

from homeassistant.components.number import NumberEntity, NumberMode
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.entity import DeviceInfo

from .const import DOMAIN, CONF_SETPOINT

import logging

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the number entities."""
    async_add_entities([VentilationSetpointNumber(hass, config_entry)])


class VentilationSetpointNumber(NumberEntity):
    """Representation of the ventilation controller setpoint."""

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry) -> None:
        """Initialize the setpoint number entity."""
        self._config_entry = config_entry
        self._hass = hass
        
        # Get current setpoint from runtime data first, then config
        domain_data = hass.data.get(DOMAIN, {})
        entry_data = domain_data.get(config_entry.entry_id, {})
        runtime_setpoint = entry_data.get("current_setpoint")
        
        if runtime_setpoint is not None:
            self._value = runtime_setpoint
        else:
            # Fallback to config if no runtime value
            cfg = {**config_entry.data, **config_entry.options}
            self._value = cfg.get(CONF_SETPOINT, 0.5)
        
        # Entity configuration
        self._attr_name = "Setpoint"
        self._attr_unique_id = f"{config_entry.entry_id}_setpoint"
        self._attr_native_min_value = 0.0
        self._attr_native_max_value = 5.0
        self._attr_native_step = 0.1
        self._attr_mode = NumberMode.SLIDER
        self._attr_native_unit_of_measurement = "index"
        self._attr_icon = "mdi:target"

    @property
    def device_info(self) -> DeviceInfo:
        """Return device information."""
        return DeviceInfo(
            identifiers={(DOMAIN, self._config_entry.entry_id)},
            name="PID Ventilation Controller",
            manufacturer="Custom",
            model="PID Controller",
        )

    @property
    def native_value(self) -> float:
        """Return the current setpoint value."""
        # Always get the most current value from runtime data
        domain_data = self._hass.data.get(DOMAIN, {})
        entry_data = domain_data.get(self._config_entry.entry_id, {})
        runtime_setpoint = entry_data.get("current_setpoint")
        
        if runtime_setpoint is not None:
            self._value = runtime_setpoint
            return runtime_setpoint
        
        return self._value

    async def async_set_native_value(self, value: float) -> None:
        """Update the setpoint value."""
        self._value = value
        
        _LOGGER.info(f"Setpoint updated to {value}")
        
        # Store the updated value in the runtime data so PID controller can use it immediately
        # Do NOT update the config entry as this can cause reload loops
        domain_data = self._hass.data.get(DOMAIN, {})
        entry_data = domain_data.get(self._config_entry.entry_id, {})
        if isinstance(entry_data, dict):
            entry_data["current_setpoint"] = value
        
        # Trigger a state update
        self.async_write_ha_state()
