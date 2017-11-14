""" Collection of helper functions used for data retrieval """


import csv
import sqlite3
from datetime import date, timedelta


DB_PATH = 'Data/corpus.db'
START_DATE = date(2001, 9, 11)
END_DATE = date(2003, 3, 19)


def generate_csv(table):
    """ Outputs a csv for the given table """
    conn = sqlite3.connect(DB_PATH)
    curs = conn.cursor()
    data = curs.execute("SELECT * FROM " + table)
    filename = table.lower() + '.csv'
    csv_file = open(filename, 'w', newline="")
    writer = csv.writer(csv_file, delimiter=';')
    writer.writerows(data)
    csv_file.close()


def generate_divisions_csv():
    """ Generates a csv of the contents of the divisions table """
    generate_csv('DIVISION')


def generate_members_csv():
    """ Generates a csv of the contents of the members table """
    generate_csv('MEMBER')


def generate_votes_csv():
    """ Generates a csv of the contents of the votes table """
    generate_csv('VOTE')


def generate_debates_csv():
    """ Generates a csv of the contents of the debates table """
    generate_csv('DEBATE')


def date_range(start_date, end_date):
    """Retuns all dates between start_date (inclusive) and end_date (exclusive)"""
    for count in range(int((end_date - start_date).days)):
        yield start_date + timedelta(count)
