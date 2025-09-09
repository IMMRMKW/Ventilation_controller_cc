#!/usr/bin/env python3
"""Test script to verify config flow schema changes."""

# Mock the Home Assistant imports for testing
class MockHomeAssistant:
    pass

class MockDeviceRegistry:
    @staticmethod
    def async_get(hass):
        return MockDeviceRegistry()
    
    def async_get(self, device_id):
        # Mock device with identifiers
        class MockDevice:
            identifiers = [("ramses_cc", "29:162275"), ("other", "short")]
        
        return MockDevice()

# Mock device_registry
import sys
import types

# Create mock modules
mock_ha_core = types.ModuleType('homeassistant.core')
mock_ha_helpers = types.ModuleType('homeassistant.helpers')
mock_device_registry = types.ModuleType('homeassistant.helpers.device_registry')

# Add to sys.modules
sys.modules['homeassistant'] = types.ModuleType('homeassistant')
sys.modules['homeassistant.core'] = mock_ha_core
sys.modules['homeassistant.helpers'] = mock_ha_helpers
sys.modules['homeassistant.helpers.device_registry'] = mock_device_registry
sys.modules['homeassistant.helpers.entity_registry'] = types.ModuleType('homeassistant.helpers.entity_registry')
sys.modules['homeassistant.helpers.selector'] = types.ModuleType('homeassistant.helpers.selector')
sys.modules['homeassistant.helpers.config_validation'] = types.ModuleType('homeassistant.helpers.config_validation')
sys.modules['homeassistant.config_entries'] = types.ModuleType('homeassistant.config_entries')
sys.modules['homeassistant.data_entry_flow'] = types.ModuleType('homeassistant.data_entry_flow')
sys.modules['homeassistant.exceptions'] = types.ModuleType('homeassistant.exceptions')
sys.modules['voluptuous'] = types.ModuleType('voluptuous')

# Add classes to mock modules
mock_ha_core.HomeAssistant = MockHomeAssistant
mock_device_registry.async_get = MockDeviceRegistry.async_get
mock_ha_helpers.device_registry = mock_device_registry

# Add necessary voluptuous classes
import types
vol_module = sys.modules['voluptuous']
vol_module.Schema = lambda x, **kwargs: x
vol_module.Required = lambda key, **kwargs: key
vol_module.Optional = lambda key, **kwargs: key
vol_module.Coerce = lambda func: func
vol_module.ALLOW_EXTRA = "allow_extra"

# Add selector classes
selector_module = sys.modules['homeassistant.helpers.selector']
selector_module.DeviceSelector = lambda config: "DeviceSelector"
selector_module.DeviceSelectorConfig = lambda **kwargs: kwargs
selector_module.SelectSelector = lambda config: "SelectSelector"  
selector_module.SelectSelectorConfig = lambda **kwargs: kwargs
selector_module.BooleanSelector = lambda: "BooleanSelector"

# Add config validation
cv_module = sys.modules['homeassistant.helpers.config_validation']
cv_module.string = str

print("Testing device selection schema...")

try:
    # Import the functions we want to test
    sys.path.append('d:/github/Ventilation_controller_cc/custom_components/ventilation')
    from config_flow import get_device_selection_schema, get_zone_config_schema
    
    # Test device selection schema
    device_schema = get_device_selection_schema()
    print("✓ Device selection schema created successfully")
    print("  Schema keys:", list(device_schema.keys()))
    
    # Test zone config schema
    zone_schema = get_zone_config_schema(1, {}, "29:162275", "32:146231")
    print("✓ Zone config schema created successfully")
    print("  Schema keys:", list(zone_schema.keys()))
    
    print("\n✅ All schema tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
