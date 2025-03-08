import pandas as pd

def process_data(csv_path):
    """
    Reads the CSV file, converts datetime columns, computes total vehicle counts,
    and calculates a congestion index normalized to a 0-100 scale.
    """
    data = pd.read_csv(csv_path)
    
    # Convert potential datetime columns
    for col in ['count_date', 'start_time', 'end_time']:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors='coerce')
    
    # Identify vehicle count columns (e.g., those containing "car", "truck", or "bus")
    vehicle_cols = [col for col in data.columns if any(x in col.lower() for x in ['car', 'truck', 'bus'])]
    if not vehicle_cols:
        print("Warning: No vehicle count columns found in the data!")
    data['total_vehicles'] = data[vehicle_cols].sum(axis=1)
    
    # Compute congestion index on a 0-100 scale
    max_val = data['total_vehicles'].max()
    if max_val == 0:
        max_val = 1  # avoid division by zero
    data['congestion_index'] = (data['total_vehicles'] / max_val) * 100
    
    return data
