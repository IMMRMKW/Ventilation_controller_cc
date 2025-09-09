# Multi-Zone Sensor Assignment Flow

## How It Works

The CO2 sensor configuration now uses a **two-phase approach** that provides the most intuitive experience possible within Home Assistant's config flow limitations.

## User Experience Flow

### Phase 1: Sensor Selection
1. User sees "CO2 Sensors" form
2. User selects sensors using entity picker (e.g., sensor.living_room_co2, sensor.bedroom_co2)
3. User clicks "Next"

### Phase 2: Zone Assignment (Multi-Zone Only)
1. **If single zone**: Continues directly to next step
2. **If multi-zone**: Shows zone assignment form with:
   - Dropdown for each selected sensor
   - Clear labels like "Living Room CO2 → Zone"
   - Zone options (Zone 1, Zone 2, etc.)
3. User assigns each sensor to appropriate zones
4. User clicks "Next" to continue

## Technical Implementation

### Key Functions

```python
get_co2_sensors_selection_schema(current_sensors=None)
# Returns: Basic sensor selection form

get_co2_sensors_zones_schema(hass, sensors, current_zones, num_zones)  
# Returns: Zone assignment form (only if needed)
```

### Smart Detection Logic

```python
# In async_step_co2_sensors():
has_zone_data = any(key.startswith("co2_sensor_") and key.endswith("_zone") 
                   for key in user_input.keys())

if has_zone_data:
    # Process zone assignments - user is done
    # Extract zones and proceed to next step
else:
    # Show zone assignment form 
    return self.async_show_form(step_id="co2_sensors", data_schema=zones_schema)
```

## Benefits

1. **Clear User Intent**: Phase 1 focuses purely on sensor selection
2. **Immediate Feedback**: Phase 2 shows all zone dropdowns at once
3. **No Empty UI**: Zone selectors only appear when needed and populated
4. **Standard HA Pattern**: Follows Home Assistant config flow conventions
5. **Consistent Behavior**: Works the same for CO2, VOC, PM, and humidity sensors

## Example Scenarios

### Single Zone Setup
- User selects 3 CO2 sensors
- All automatically assigned to Zone 1 
- Proceeds directly to VOC sensors

### Dual Zone Setup  
- User selects 3 CO2 sensors → clicks Next
- Form reloads showing:
  - "Living Room CO2 → Zone" dropdown (Zone 1, Zone 2)
  - "Bedroom CO2 → Zone" dropdown (Zone 1, Zone 2) 
  - "Office CO2 → Zone" dropdown (Zone 1, Zone 2)
- User assigns zones → clicks Next → continues to VOC sensors

### Large Multi-Zone Setup
- User selects 5 CO2 sensors → clicks Next
- Shows 5 zone dropdowns (Zone 1-4 options)
- User assigns each sensor to appropriate zone
- Clean, organized assignment process

## Next Steps

This same pattern will be applied to:
- VOC sensor assignment
- PM sensor assignment  
- Humidity sensor assignment

Each sensor type gets its own two-phase configuration for consistent UX.
