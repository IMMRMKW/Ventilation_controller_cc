from __future__ import annotations

from homeassistant.components.switch import SwitchEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DOMAIN

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback) -> None:
    async_add_entities([VentilationEnableSwitch(hass, entry)])


class VentilationEnableSwitch(SwitchEntity):
    _attr_should_poll = False
    _attr_has_entity_name = True

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self.entry = entry
        self._attr_name = "Enabled"
        self._attr_unique_id = f"{entry.entry_id}_enabled"
        self._attr_icon = "mdi:fan"

    @property
    def is_on(self) -> bool:
        data = self.hass.data.get(DOMAIN, {}).get(self.entry.entry_id)
        if isinstance(data, dict):
            return data.get("enabled", True)
        return True

    async def async_turn_on(self, **kwargs) -> None:
        domain = self.hass.data.setdefault(DOMAIN, {})
        entry_data = domain.setdefault(self.entry.entry_id, {})
        entry_data["enabled"] = True
        self.async_write_ha_state()

    async def async_turn_off(self, **kwargs) -> None:
        domain = self.hass.data.setdefault(DOMAIN, {})
        entry_data = domain.setdefault(self.entry.entry_id, {})
        entry_data["enabled"] = False
        self.async_write_ha_state()

    @property
    def device_info(self):
        return {
            "identifiers": {(DOMAIN, self.entry.entry_id)},
            "name": "PID Ventilation",
            "manufacturer": "Custom",
        }
