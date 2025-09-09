# Separate Step IDs for Zone Assignment

## Problem Solved

The issue was that both sensor selection and zone assignment phases used the same step ID and translations, causing confusing titles and descriptions. When users were assigning zones, they still saw "CO2 Sensors" title with description saying "you'll assign sensors to zones in the next step" - but they were already in that step!

## Solution Implemented

### 1. Separate Step IDs

**Phase 1 - Sensor Selection:**
- Step ID: `co2_sensors` 
- Title: "CO2 Sensors"
- Description: "Select your CO2 sensors. For multi-zone setups, you'll assign sensors to zones in the next step."

**Phase 2 - Zone Assignment:**  
- Step ID: `co2_sensors_zones`
- Title: "CO2 Sensors - Zone Assignment"
- Description: "Assign each CO2 sensor to the zone it monitors. Each sensor can monitor a different zone in your ventilation system."

### 2. Dedicated Step Handlers

Added `async_step_co2_sensors_zones()` handler that:
- Handles zone assignment form submission
- Processes zone assignments for each sensor
- Provides clear back navigation to sensor selection
- Shows sensor-specific zone dropdowns with friendly names

### 3. Clear User Flow

```
User Journey (Multi-Zone):
1. "CO2 Sensors" → Select sensors → Click "Next"
2. "CO2 Sensors - Zone Assignment" → Assign zones → Click "Next"  
3. Continue to next sensor type

User Journey (Single Zone):
1. "CO2 Sensors" → Select sensors → Click "Next"
2. Skip zone assignment (auto-assigned to Zone 1)
3. Continue to next sensor type
```

### 4. Translation Structure

```json
{
  "co2_sensors": {
    "title": "CO2 Sensors",
    "description": "Select your CO2 sensors. For multi-zone setups, you'll assign sensors to zones in the next step.",
    "data": {
      "co2_sensors": "CO2 Sensor Entities",
      "back": "Back"
    }
  },
  "co2_sensors_zones": {
    "title": "CO2 Sensors - Zone Assignment", 
    "description": "Assign each CO2 sensor to the zone it monitors. Each sensor can monitor a different zone in your ventilation system.",
    "data": {
      "co2_sensor_0_zone": "{sensor_0_name} → Zone",
      "co2_sensor_1_zone": "{sensor_1_name} → Zone"
    }
  }
}
```

### 5. Applied to All Sensor Types

Created separate step IDs and translations for:
- **CO2**: `co2_sensors` → `co2_sensors_zones` ✅ 
- **VOC**: `voc_sensors` → `voc_sensors_zones` ✅ 
- **PM**: `pm_sensors` → `pm_sensors_zones` ✅
- **Humidity**: `humidity_sensors` → `humidity_sensors_zones` ✅

## Benefits

### Before:
- **Confusing**: "CO2 Sensors" title during zone assignment
- **Wrong Context**: "you'll assign sensors to zones in the next step" when already in that step
- **Poor UX**: Users didn't know what stage they were in

### After:
- **Clear Purpose**: "CO2 Sensors - Zone Assignment" clearly indicates current task
- **Accurate Description**: "Assign each CO2 sensor to the zone it monitors"
- **Contextual**: Users know exactly what they're doing at each step
- **Professional**: Matches Home Assistant's standard multi-step patterns

## Technical Implementation

### Config Flow Changes:
```python
# Phase 1: Sensor selection 
return self.async_show_form(step_id="co2_sensors", ...)

# Phase 2: Zone assignment
return self.async_show_form(step_id="co2_sensors_zones", ...)
```

### Back Navigation:
```python
# From zone assignment back to sensor selection
if user_input.get("back"):
    return await self.async_step_co2_sensors()
```

### Sensor Name Display:
```python
# Zone labels show actual sensor names
"co2_sensor_0_zone": "{sensor_0_name} → Zone"
```

## Result

Users now have a **crystal clear understanding** of where they are in the configuration process:

1. **"CO2 Sensors"** - I'm selecting which CO2 sensors to use
2. **"CO2 Sensors - Zone Assignment"** - I'm assigning these sensors to zones  
3. **"VOC Sensors"** - I'm selecting which VOC sensors to use
4. **"VOC Sensors - Zone Assignment"** - I'm assigning these sensors to zones

Each step has contextually appropriate titles, descriptions, and field labels that match what the user is actually doing at that moment.
