from __future__ import annotations

from homeassistant.components.sensor import SensorEntity, SensorDeviceClass, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.dispatcher import async_dispatcher_connect

from .const import DOMAIN, SIGNAL_AQI_UPDATED, SIGNAL_RATE_UPDATED

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback) -> None:
    async_add_entities([
        AirQualityIndexSensor(hass, entry),
        FanRatePercentSensor(hass, entry),
    ])


class AirQualityIndexSensor(SensorEntity):
    _attr_should_poll = False
    _attr_has_entity_name = True
    _attr_native_unit_of_measurement = None
    _attr_device_class = None
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:gauge"
    _attr_suggested_display_precision = 2

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self.entry = entry
        self._attr_name = "Air Quality Index"
        self._attr_unique_id = f"{entry.entry_id}_aqi"
        # initial value if already present
        data = hass.data.get(DOMAIN, {}).get(entry.entry_id)
        initial = data.get("aqi", 0.0) if isinstance(data, dict) else 0.0
        try:
            self._attr_native_value = round(float(initial), 2)
        except Exception:
            self._attr_native_value = 0.0
        self._unsub = None

    async def async_added_to_hass(self) -> None:
        @callback
        def _handle_update(payload: dict) -> None:
            try:
                self._attr_native_value = round(float(payload.get("aqi", 0.0)), 2)
            except Exception:
                self._attr_native_value = 0.0
            self.async_write_ha_state()

        self._unsub = async_dispatcher_connect(
            self.hass, f"{SIGNAL_AQI_UPDATED}_{self.entry.entry_id}", _handle_update
        )

    async def async_will_remove_from_hass(self) -> None:
        if self._unsub:
            self._unsub()
            self._unsub = None

    @property
    def device_info(self):
        return {
            "identifiers": {(DOMAIN, self.entry.entry_id)},
            "name": "PID Ventilation",
            "manufacturer": "Custom",
        }


class FanRatePercentSensor(SensorEntity):
    _attr_should_poll = False
    _attr_has_entity_name = True
    _attr_native_unit_of_measurement = "%"
    _attr_device_class = None
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:fan"
    _attr_suggested_display_precision = 0

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        self.hass = hass
        self.entry = entry
        self._attr_name = "Fan Rate"
        self._attr_unique_id = f"{entry.entry_id}_rate_pct"
        data = hass.data.get(DOMAIN, {}).get(entry.entry_id)
        initial = data.get("rate_pct", 0.0) if isinstance(data, dict) else 0.0
        try:
            self._attr_native_value = round(float(initial), 0)
        except Exception:
            self._attr_native_value = 0.0
        self._unsub = None

    async def async_added_to_hass(self) -> None:
        @callback
        def _handle_update(payload: dict) -> None:
            try:
                self._attr_native_value = round(float(payload.get("rate_pct", 0.0)), 0)
            except Exception:
                self._attr_native_value = 0.0
            self.async_write_ha_state()

        self._unsub = async_dispatcher_connect(
            self.hass, f"{SIGNAL_RATE_UPDATED}_{self.entry.entry_id}", _handle_update
        )

    async def async_will_remove_from_hass(self) -> None:
        if self._unsub:
            self._unsub()
            self._unsub = None

    @property
    def device_info(self):
        return {
            "identifiers": {(DOMAIN, self.entry.entry_id)},
            "name": "PID Ventilation",
            "manufacturer": "Custom",
        }
