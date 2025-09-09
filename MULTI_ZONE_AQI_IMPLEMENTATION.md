# Multi-Zone Air Quality Implementation

## Overview

Successfully implemented **per-zone air quality sensors** that calculate separate AQI values for each zone based on the sensors assigned to that zone during configuration.

## Key Changes Made

### 1. Configuration Constants (const.py)
```python
CONF_CO2_SENSOR_ZONES = "co2_sensor_zones"
CONF_VOC_SENSOR_ZONES = "voc_sensor_zones" 
CONF_PM_SENSOR_ZONES = "pm_sensor_zones"
CONF_HUMIDITY_SENSOR_ZONES = "humidity_sensor_zones"
```

### 2. Multi-Zone Sensor Creation (sensor.py)
```python
async def async_setup_entry(hass, entry, async_add_entities):
    # Determine number of zones from valve devices
    valve_devices = cfg.get(CONF_VALVE_DEVICES, [])
    num_zones = max(1, len(valve_devices))
    
    # Create one AQI sensor per zone
    for zone_id in range(1, num_zones + 1):
        sensors.append(AirQualityIndexSensor(hass, entry, zone_id))
```

**Results in entities like:**
- `sensor.pid_ventilation_air_quality_index_zone_1`
- `sensor.pid_ventilation_air_quality_index_zone_2`  
- `sensor.pid_ventilation_air_quality_index_zone_3`

### 3. Zone-Aware Air Quality Calculation (__init__.py)
```python
def _calculate_zone_air_quality(hass, zone_id, sensors, zones, ...):
    # Filter sensors assigned to this zone
    zone_co2_sensors = [sensor for i, sensor in enumerate(co2_sensors) 
                       if i < len(co2_zones) and co2_zones[i] == zone_id]
    # ... similar for VOC, PM, humidity
    
    # Calculate AQI using only this zone's sensors
    return _worst_air_quality_index(hass, zone_sensors, ...)
```

### 4. Multi-Zone PID Control Loop
```python
# Calculate AQI per zone and find worst overall
zone_aqi_data = {}
worst_zone_aqi = 0.0

for zone_id in range(1, num_zones + 1):
    zone_aqi, ... = _calculate_zone_air_quality(hass, zone_id, ...)
    zone_aqi_data[zone_id] = zone_aqi
    
    # Track worst zone for overall fan control
    if zone_aqi > worst_zone_aqi:
        worst_zone_aqi = zone_aqi
        worst_zone = zone_id

# Use worst zone AQI for fan control        
air_quality_index = worst_zone_aqi
```

## How It Works

### Example Configuration Scenario:

**Zones:** 3 zones (based on 3 valve devices)

**Sensor Assignments:**
- Living Room CO2 → Zone 1
- Bedroom CO2 → Zone 2  
- Office CO2 → Zone 3
- Kitchen Humidity → Zone 1
- Bathroom Humidity → Zone 2

### Runtime Behavior:

**Zone 1 (Living Room + Kitchen):**
- Uses: Living Room CO2 + Kitchen Humidity
- Calculates: Zone 1 AQI = max(co2_index, humidity_index)
- Entity: `sensor.pid_ventilation_air_quality_index_zone_1`

**Zone 2 (Bedroom + Bathroom):**
- Uses: Bedroom CO2 + Bathroom Humidity  
- Calculates: Zone 2 AQI = max(co2_index, humidity_index)
- Entity: `sensor.pid_ventilation_air_quality_index_zone_2`

**Zone 3 (Office):**
- Uses: Office CO2 only
- Calculates: Zone 3 AQI = co2_index  
- Entity: `sensor.pid_ventilation_air_quality_index_zone_3`

**Overall System Control:**
- Compares all zone AQIs: Zone 1=1.2, Zone 2=2.8, Zone 3=0.5
- Uses worst zone (Zone 2, AQI=2.8) for fan speed control
- All zones get fresh air, but speed based on worst-performing zone

## Benefits

### 1. **Zone-Specific Monitoring**
Users can see individual zone air quality:
- "Living room is excellent (0.5), but bedroom needs attention (2.8)"
- Create zone-specific automations and alerts
- Track air quality patterns per room

### 2. **Intelligent Control Strategy** 
- System responds to worst-performing zone
- Ensures no zone is neglected
- Prevents over-ventilation when only one zone has issues

### 3. **Scalable Architecture**
- Works with 1-N zones automatically
- Single zone: Traditional single AQI sensor  
- Multi-zone: One AQI sensor per zone
- No configuration changes needed for different zone counts

### 4. **Rich Logging**
```
PID: Zones(Zone 1: 1.20, Zone 2: 2.80, Zone 3: 0.50), 
     Worst=Zone 2 (sensor.bedroom_co2, dc=carbon_dioxide, val=1250), 
     Overall AQI=2.80, Setpoint=1.50, Error=1.30, Fan=65%, Integral=2.45
```

## Use Cases

### Home Automation Examples:

```yaml
# Zone-specific low air quality alert
- alias: "Bedroom Air Quality Alert"  
  trigger:
    platform: numeric_state
    entity_id: sensor.pid_ventilation_air_quality_index_zone_2
    above: 2.5
  action:
    - service: notify.mobile_app
      data:
        message: "Bedroom air quality is poor ({{ states('sensor.pid_ventilation_air_quality_index_zone_2') }})"

# Zone comparison dashboard
- alias: "Daily Air Quality Report"
  trigger:
    platform: time
    at: "08:00:00"
  action:
    - service: notify.home_assistant
      data:
        message: |
          Daily Air Quality:
          Living Room: {{ states('sensor.pid_ventilation_air_quality_index_zone_1') }}
          Bedroom: {{ states('sensor.pid_ventilation_air_quality_index_zone_2') }}  
          Office: {{ states('sensor.pid_ventilation_air_quality_index_zone_3') }}
```

### Dashboard Cards:
```yaml
# Multi-zone air quality gauge
type: gauge
entities:
  - entity: sensor.pid_ventilation_air_quality_index_zone_1
    name: Living Room
  - entity: sensor.pid_ventilation_air_quality_index_zone_2  
    name: Bedroom
  - entity: sensor.pid_ventilation_air_quality_index_zone_3
    name: Office
min: 0
max: 5
```

## Implementation Status

✅ **Multi-zone sensor creation** - Creates one AQI sensor per zone  
✅ **Zone-aware air quality calculation** - Filters sensors by zone assignment  
✅ **Intelligent fan control** - Uses worst zone for system control  
✅ **Rich state management** - Tracks per-zone and overall AQI data  
✅ **Comprehensive logging** - Shows zone breakdown and worst zone info  
✅ **Backward compatibility** - Works with single zone setups  

The integration now provides **granular, zone-specific air quality monitoring** while maintaining intelligent system-wide control based on the zone that needs the most attention.
