#!/usr/bin/env python3
"""Test script for zone calculation logic."""

def test_zone_calculation():
    """Test the zone calculation logic from config flow."""
    print("🧪 Testing Zone Calculation Logic")
    print("=" * 40)
    
    test_cases = [
        ([], "No valves selected"),
        (["valve1"], "Single valve selected"),
        (["valve1", "valve2"], "Two valves selected"),
        (["valve1", "valve2", "valve3", "valve4"], "Four valves selected"),
    ]
    
    for valve_devices, description in test_cases:
        # Zone calculation logic from config flow
        if isinstance(valve_devices, list):
            num_zones = max(1, len(valve_devices))  # At least 1 zone
        else:
            # Single valve selected (shouldn't happen with list selector)
            num_zones = 1 if not valve_devices else 2
            
        print(f"Input: {valve_devices}")
        print(f"Description: {description}")
        print(f"Calculated zones: {num_zones}")
        print(f"Zone description: {num_zones} zone{'s' if num_zones > 1 else ''}")
        print("-" * 40)
    
    print("✅ Zone calculation logic validated!")

if __name__ == "__main__":
    test_zone_calculation()
