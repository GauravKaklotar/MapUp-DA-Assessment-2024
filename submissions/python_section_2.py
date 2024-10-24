import pandas as pd
from datetime import time

def calculate_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Extract unique toll location ids
    toll_locations = pd.concat([df['id_start'], df['id_end']]).unique()
    toll_locations.sort()  # Sort for consistent ordering
    
    # Create an empty DataFrame for the distance matrix
    distance_matrix = pd.DataFrame(0, index=toll_locations, columns=toll_locations)
    
    # Fill in the direct distances from the input DataFrame
    for _, row in df.iterrows():
        start, end, distance = row['id_start'], row['id_end'], row['distance']
        distance_matrix.loc[start, end] = distance
        distance_matrix.loc[end, start] = distance 
    
    return distance_matrix


def unroll_distance_matrix(df) -> pd.DataFrame:
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame): A DataFrame representing the distance matrix with IDs as both rows and columns.

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    unrolled_data = []
    
    # Iterate over the DataFrame to get the distance values between each pair of IDs
    for id_start in df.index:
        for id_end in df.columns:
            if id_start != id_end:  # Exclude self-distances
                unrolled_data.append({
                    'id_start': id_start,
                    'id_end': id_end,
                    'distance': df.loc[id_start, id_end]
                })
    
    # Convert the unrolled data into a DataFrame
    unrolled_df = pd.DataFrame(unrolled_data)
    
    return unrolled_df

def find_ids_within_ten_percentage_threshold(df, reference_id) -> pd.DataFrame:
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame): DataFrame created in Question 10 containing 'id_start', 'id_end', and 'distance'.
        reference_id (int): The reference ID from which to calculate the 10% threshold.

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Calculate the average distance for the reference_id
    reference_df = df[df['id_start'] == reference_id]
    reference_avg_distance = reference_df['distance'].mean()

    # Calculate 10% of the reference average distance
    lower_bound = reference_avg_distance * 0.90  # 10% below the reference average
    upper_bound = reference_avg_distance * 1.10  # 10% above the reference average

    # Calculate average distances for all other IDs in id_start
    avg_distances = df.groupby('id_start')['distance'].mean()

    # Find the IDs whose average distance falls within the 10% threshold
    ids_within_threshold = avg_distances[(avg_distances >= lower_bound) & (avg_distances <= upper_bound)].index.tolist()

    # Return the IDs as a sorted list
    return sorted(ids_within_threshold)


def calculate_toll_rate(df) -> pd.DataFrame:
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame): The unrolled DataFrame containing 'id_start', 'id_end', and 'distance'.

    Returns:
        pandas.DataFrame: Updated DataFrame with toll rates for different vehicle types.
    """
    # Define rate coefficients for each vehicle type
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    # Calculate toll rates by multiplying distance with respective rate coefficients
    df['moto'] = df['distance'] * rate_coefficients['moto']
    df['car'] = df['distance'] * rate_coefficients['car']
    df['rv'] = df['distance'] * rate_coefficients['rv']
    df['bus'] = df['distance'] * rate_coefficients['bus']
    df['truck'] = df['distance'] * rate_coefficients['truck']

    return df


def calculate_time_based_toll_rates(df) -> pd.DataFrame:
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: DataFrame with time-based toll rates adjusted by time intervals and days of the week.
    """
    # Prepare list to store all rows to concatenate later
    rows = []

    # Define time intervals and discount factors for weekdays
    time_intervals_weekday = [
        (time(0, 0, 0), time(10, 0, 0), 0.8),  
        (time(10, 0, 0), time(18, 0, 0), 1.2),
        (time(18, 0, 0), time(23, 59, 59), 0.8) 
    ]

    # Define discount factor for weekends
    weekend_discount_factor = 0.7
    
    # Define the days of the week
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    weekends = ["Saturday", "Sunday"]

    for index, row in df.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']

        # Apply time-based tolls for weekdays
        for day in weekdays:
            for start_time, end_time, discount_factor in time_intervals_weekday:
                # Create a copy of the row to apply discounts
                toll_row = row.copy()

                # Add time and day columns
                toll_row['start_day'] = day
                toll_row['end_day'] = day
                toll_row['start_time'] = start_time
                toll_row['end_time'] = end_time

                # Apply discount to the toll rates
                toll_row['moto'] *= discount_factor
                toll_row['car'] *= discount_factor
                toll_row['rv'] *= discount_factor
                toll_row['bus'] *= discount_factor
                toll_row['truck'] *= discount_factor

                # Add the adjusted row to the list
                rows.append(toll_row)

        # Apply time-based tolls for weekends
        for day in weekends:
            # Create a copy of the row to apply weekend discounts
            toll_row = row.copy()

            toll_row['start_day'] = day
            toll_row['end_day'] = day
            toll_row['start_time'] = time(0, 0, 0)
            toll_row['end_time'] = time(23, 59, 59)

            # Apply weekend discount to the toll rates
            toll_row['moto'] *= weekend_discount_factor
            toll_row['car'] *= weekend_discount_factor
            toll_row['rv'] *= weekend_discount_factor
            toll_row['bus'] *= weekend_discount_factor
            toll_row['truck'] *= weekend_discount_factor

            # Add the adjusted row to the list
            rows.append(toll_row)

    # Concatenate all rows into a final DataFrame
    result_df = pd.concat([pd.DataFrame([row]) for row in rows], ignore_index=True)

    return result_df


# df = pd.read_csv("../datasets/dataset-2.csv")

# # Question 9: Distance Matrix Calculation
# result_1 = calculate_distance_matrix(df)
# print(result_1)


# # Question 10: Unroll Distance Matrix
# result_2 = unroll_distance_matrix(result_1)
# print(result_2)


# # Question 11: Finding IDs within Percentage Threshold
# result_3 = find_ids_within_ten_percentage_threshold(result_2, 1001400)
# print(result_3)


# # Question 12: Calculate Toll Rate
# result_4 = calculate_toll_rate(result_2)
# print(result_4)


# # Question 13: Calculate Time-Based Toll Rates
# result_5 = calculate_time_based_toll_rates(result_4)
# print(result_5)