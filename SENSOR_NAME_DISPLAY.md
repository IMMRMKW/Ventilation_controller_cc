# Sensor Name Display in Zone Selectors

## Problem Solved

Previously, zone selector labels showed generic names like "Sensor 1 → Zone", "Sensor 2 → Zone", which made it impossible for users to know which actual sensor they were assigning to which zone.

## Solution Implemented

### 1. Dynamic Translation Placeholders

Updated translation strings to use placeholders:
```json
"co2_sensor_0_zone": "{sensor_0_name} → Zone",
"co2_sensor_1_zone": "{sensor_1_name} → Zone",
```

### 2. Sensor Name Extraction

In the config flow, extract actual sensor names and pass them as placeholders:
```python
# Add sensor names for translation placeholders
for i, sensor in enumerate(selected_sensors):
    sensor_name = get_entity_display_name(self.hass, sensor)
    placeholders[f"sensor_{i}_name"] = sensor_name

return self.async_show_form(
    step_id="co2_sensors", 
    data_schema=zones_schema,
    description_placeholders=placeholders
)
```

### 3. New Schema Functions

Created dedicated selection and zone assignment schemas for all sensor types:

**Selection Schemas (Phase 1):**
- `get_co2_sensors_selection_schema()`
- `get_voc_sensors_selection_schema()`  
- `get_pm_sensors_selection_schema()`
- `get_humidity_sensors_selection_schema()`

**Zone Assignment Schemas (Phase 2):**
- `get_co2_sensors_zones_schema()`
- `get_voc_sensors_zones_schema()`
- `get_pm_sensors_zones_schema()` 
- `get_humidity_sensors_zones_schema()`

### 4. Updated Translations

All sensor types now use the same consistent pattern:
```json
"voc_sensor_0_zone": "{sensor_0_name} → Zone",
"pm_sensor_0_zone": "{sensor_0_name} → Zone", 
"humidity_sensor_0_zone": "{sensor_0_name} → Zone",
```

## User Experience

### Before:
- "CO2 Sensor 1 → Zone" (dropdown)
- "CO2 Sensor 2 → Zone" (dropdown)

### After:
- "Living Room CO2 → Zone" (dropdown)
- "Bedroom Air Quality → Zone" (dropdown)

## Example Scenario

User has these sensors:
- `sensor.living_room_co2` (friendly name: "Living Room CO2")
- `sensor.bedroom_air_quality` (friendly name: "Bedroom Air Quality")
- `sensor.office_co2_ppm` (friendly name: "Office CO2 PPM")

Zone assignment form shows:
- **Living Room CO2 → Zone** [Zone 1 ▼]
- **Bedroom Air Quality → Zone** [Zone 2 ▼]  
- **Office CO2 PPM → Zone** [Zone 1 ▼]

## Benefits

1. **Clear Identification**: Users see actual sensor names, not generic numbers
2. **No Memory Required**: No need to remember which "Sensor 1" corresponds to which actual device
3. **Consistent Pattern**: Works across all sensor types (CO2, VOC, PM, Humidity)
4. **Home Assistant Standard**: Uses HA's standard translation placeholder system
5. **Future-Proof**: Works with any number of sensors and any sensor naming convention

## Implementation Status

✅ **Completed for CO2 sensors** - Full two-phase flow with sensor name display
🔄 **Ready for Other Types** - Schema functions created, need to update step functions

## Next Steps

Apply the two-phase pattern with sensor name display to:
- VOC sensor step (`async_step_voc_sensors`)
- PM sensor step (`async_step_pm_sensors`)  
- Humidity sensor step (`async_step_humidity_sensors`)

This will provide consistent, user-friendly zone assignment across all sensor types.
