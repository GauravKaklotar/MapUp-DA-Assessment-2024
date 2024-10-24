from typing import Dict, List
from itertools import permutations
import polyline
import pandas as pd
from math import radians, cos, sin, sqrt, atan2
import re


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    # Your code goes here.
    result = []
    for i in range(0, len(lst), n):
        group = []
        for j in range(min(n, len(lst) - i)):
            group.insert(0, lst[i + j])
        result.extend(group)
    return result


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    length_dict = {}
    for word in lst:
        length = len(word)
        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(word)
    
    # Sorting the dictionary by length of the words
    return dict(sorted(length_dict.items()))

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    flattened = {}

    def _flatten(current_dict, parent_key):
        for k, v in current_dict.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                _flatten(v, new_key)
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    _flatten({f"{k}[{i}]": item}, new_key)
            else:
                flattened[new_key] = v

    _flatten(nested_dict, parent_key='')
    return flattened

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    return list(map(list, set(permutations(nums))))


def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    date_patterns = [
        r'\b\d{2}-\d{2}-\d{4}\b',  # dd-mm-yyyy
        r'\b\d{2}/\d{2}/\d{4}\b',  # mm/dd/yyyy
        r'\b\d{4}\.\d{2}\.\d{2}\b'  # yyyy.mm.dd
    ]
    
    dates = []
    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        dates.extend(matches)
    
    return dates

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    
    def haversine(lat1, lon1, lat2, lon2):
        # Radius of the Earth in meters
        R = 6371000
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c

    try:
        coordinates = polyline.decode(polyline_str)
    except Exception as e:
        print(f"Error decoding polyline: {e}")
        return pd.DataFrame(columns=['latitude', 'longitude', 'distance'])  # Return empty DataFrame

    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    df['distance'] = 0.0

    for i in range(1, len(df)):
        df.loc[i, 'distance'] = haversine(df.loc[i-1, 'latitude'], df.loc[i-1, 'longitude'],
                                          df.loc[i, 'latitude'], df.loc[i, 'longitude'])

    return df


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    n = len(matrix)

    # Rotate matrix 90 degrees clockwise
    rotated_matrix = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]

    # Create a new matrix for the final result
    result_matrix = [[0] * n for _ in range(n)]

    # Calculate the transformed matrix
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]
            result_matrix[i][j] = row_sum + col_sum

    return result_matrix


def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period.

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Convert start and end timestamps to datetime
    df['start_timestamp'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'], errors='coerce')
    df['end_timestamp'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'], errors='coerce')

    # Group by (id, id_2) and calculate the time coverage for each pair
    def check_completeness(group): 
        unique_days = set(group['startDay'].unique()).union(set(group['endDay'].unique()))
        if len(unique_days) < 7:
            return False

        # Check for 24-hour coverage
        time_covered = (group['end_timestamp'].max() - group['start_timestamp'].min())
        return time_covered.total_seconds() >= 24 * 3600  # Must cover at least 24 hours

    result = df.groupby(['id', 'id_2']).apply(check_completeness)
    return result


# # Question 1: Reverse List by N Elements
# print(reverse_by_n_elements([1, 2, 3, 4, 5, 6, 7, 8], 3))


# # Question 2: Lists & Dictionaries
# print(group_by_length(["apple", "bat", "car", "elephant", "dog", "bear"]))


# # Question 3: Flatten a Nested Dictionary
# print(flatten_dict({
#     "road": {
#         "name": "Highway 1",
#         "length": 350,
#         "sections": [
#             {
#                 "id": 1,
#                 "condition": {
#                     "pavement": "good",
#                     "traffic": "moderate"
#                 }
#             }
#         ]
#     }
# }))


# # Question 4: Generate Unique Permutations
# print(unique_permutations([1, 1, 2]))


# # Question 5: Find All Dates in a Text
# print(find_all_dates("I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."))


# # Question 6: Decode Polyline, Convert to DataFrame with Distances
# print(polyline_to_dataframe('u{~vF{w~j~I'))


# # Question 7: Matrix Rotation and Transformation
# print(rotate_and_multiply_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))


# # Question 8: Time Check
# df = pd.read_csv("../datasets/dataset-1.csv")
# print(time_check(df))
