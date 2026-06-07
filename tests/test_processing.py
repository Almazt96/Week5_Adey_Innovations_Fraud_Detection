import pytest
import pandas as pd
import numpy as np
from src.processing import FraudDataPipeline

def test_ip_to_int_conversion():
    """Validates that standard IPv4 formats resolve correctly to their math equivalents."""
    pipeline = FraudDataPipeline(raw_data_dir="", processed_data_dir="")
    sample_ips = pd.Series(["0.0.0.5", "10.0.0.1"])
    
    # Expected results computed explicitly
    # 0.0.0.5 = 5
    # 10.0.0.1 = (10 << 24) + 1 = 167772161
    expected_outputs = [5, 167772161]
    
    resolved = pipeline.ip_to_int(sample_ips)
    assert resolved.iloc[0] == expected_outputs[0]
    assert resolved.iloc[1] == expected_outputs[1]

def test_merge_geolocation_bounds():
    """Tests if our range-based merge accurately flags IPs out of boundary rules."""
    pipeline = FraudDataPipeline(raw_data_dir="", processed_data_dir="")
    
    # Setup structural dummy frames
    fraud_data = pd.DataFrame({'ip_address': [15, 50], 'user_id': [1, 2]})
    ip_map = pd.DataFrame({
        'lower_bound_ip_address': [10],
        'upper_bound_ip_address': [20],
        'country': ['TestCountry']
    })
    
    output = pipeline.merge_geolocation(fraud_data, ip_map)
    
    # 15 falls inside [10, 20] -> TestCountry
    # 50 falls completely outside ranges -> Unknown
    assert output.loc[output['user_id'] == 1, 'country'].values[0] == 'TestCountry'
    assert output.loc[output['user_id'] == 2, 'country'].values[0] == 'Unknown'