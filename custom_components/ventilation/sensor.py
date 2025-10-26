from __future__ import annotations

import logging
from homeassistant.components.sensor import SensorEntity, SensorDeviceClass, SensorStateClass
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.dispatcher import async_dispatcher_connect

from .const import DOMAIN, SIGNAL_AQI_UPDATED, SIGNAL_RATE_UPDATED, CONF_ZONE_CONFIGS, CONF_NUM_ZONES

_LOGGER = logging.getLogger(__name__)

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback) -> None:
    # Import the improved merge function to prevent sensor duplication bugs
    from .config_flow import merge_config_with_validation
    
    # Get properly merged and validated configuration
    cfg = merge_config_with_validation(entry.data, entry.options, allow_cleanup=False)
    
    # Get zone configurations and determine number of zones (now guaranteed consistent)
    zone_configs = cfg.get(CONF_ZONE_CONFIGS, {})
    num_zones = len(zone_configs) if zone_configs else int(cfg.get(CONF_NUM_ZONES, 1))
    
    _LOGGER.debug(f"Sensor setup: {num_zones} zones from validated config")
    
    # Create sensors list
    sensors = []
    
    # Create one AQI sensor per zone
    for zone_id in range(1, num_zones + 1):
        sensors.append(AirQualityIndexSensor(hass, entry, zone_id))
        _LOGGER.debug(f"Created AQI sensor for zone {zone_id}")
    
    # Add single fan rate sensor (global)
    sensors.append(FanRatePercentSensor(hass, entry))
    
    _LOGGER.info(f"Setup complete: {num_zones} AQI sensors + 1 fan rate sensor = {len(sensors)} total sensors")
    async_add_entities(sensors)


class AirQualityIndexSensor(SensorEntity):
    _attr_should_poll = False
    _attr_has_entity_name = True
    _attr_native_unit_of_measurement = None
    _attr_device_class = None
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:gauge"
    _attr_suggested_display_precision = 2

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry, zone_id: int = None) -> None:
        self.hass = hass
        self.entry = entry
        self.zone_id = zone_id
        
        # Set name and unique_id based on zone
        if zone_id is not None:
            self._attr_name = f"Air Quality Index Zone {zone_id}"
            self._attr_unique_id = f"{entry.entry_id}_aqi_zone_{zone_id}"
        else:
            # Fallback for single zone
            self._attr_name = "Air Quality Index"
            self._attr_unique_id = f"{entry.entry_id}_aqi"
        
        # Get initial value if already present
        data = hass.data.get(DOMAIN, {}).get(entry.entry_id)
        if zone_id is not None:
            zone_data = data.get("zone_aqi", {}) if isinstance(data, dict) else {}
            initial = zone_data.get(zone_id, 0.0)
        else:
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
                if self.zone_id is not None:
                    # Zone-specific AQI
                    zone_aqi_data = payload.get("zone_aqi", {})
                    aqi_value = zone_aqi_data.get(self.zone_id, 0.0)
                else:
                    # Global AQI (fallback)
                    aqi_value = payload.get("aqi", 0.0)
                    
                self._attr_native_value = round(float(aqi_value), 2)
            except Exception:
                self._attr_native_value = 0.0
            self.async_write_ha_state()

        # Listen for AQI updates (single signal with all zone data)
        signal_name = f"{SIGNAL_AQI_UPDATED}_{self.entry.entry_id}"
        self._unsub = async_dispatcher_connect(
            self.hass, signal_name, _handle_update
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
