""" Collection of helper functions used for data retrieval """

from datetime import timedelta

def date_range(start_date, end_date):
    """Retuns all dates between start_date (inclusive) and end_date (exclusive)"""
    for count in range(int((end_date - start_date).days)):
        yield start_date + timedelta(count)