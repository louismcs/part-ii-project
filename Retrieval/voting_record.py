""" Creates a database of MPs and their votes """


import json
import csv
import sqlite3
from datetime import date
import requests
from helper import date_range


DB_PATH = 'Data/Corpus/db_test21.db'


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


def create_tables():
    """ Creates the Member, Division and Vote tables """
    conn = sqlite3.connect(DB_PATH)
    curs = conn.cursor()

    curs.execute('''CREATE TABLE MEMBER
            (ID                TEXT   PRIMARY KEY   NOT NULL,
             FULL_NAME         TEXT,
             GIVEN_NAME        TEXT,
             ADDITIONAL_NAME   TEXT,
             FAMILY_NAME       TEXT,
             PARTY             TEXT,
             CONSTITUENCY      TEXT);''')
    conn.commit()

    curs.execute('''CREATE TABLE DIVISION
            (ID      TEXT   PRIMARY KEY   NOT NULL,
             DATE    TEXT,
             TITLE   TEXT);''')
    conn.commit()

    curs.execute('''CREATE TABLE VOTE
            (MEMBER_ID     TEXT   NOT NULL,
             DIVISION_ID   TEXT   NOT NULL,
             VOTE          TEXT,
             PRIMARY KEY(MEMBER_ID, DIVISION_ID),
             FOREIGN KEY(MEMBER_ID)   REFERENCES MEMBER(ID),
             FOREIGN KEY(DIVISION_ID) REFERENCES DIVISION(ID));''')
    conn.commit()

    conn.close()


def get_division_id(about):
    """ Returns the division id given the about field from the division json """
    return about[36:]


def division_inserts(day):
    """ Inserts all the divisions for a given day into the database """
    division_date = day.strftime('%Y-%m-%d')
    url = 'http://lda.data.parliament.uk/commonsdivisions.json?date=' \
           + division_date \
           + '&exists-date=true&_view=Commons+Divisions&_pageSize=500&_page=0'
    res = requests.get(url)
    obj = json.loads(res.text)
    divisions = obj['result']['items']
    conn = sqlite3.connect(DB_PATH)
    curs = conn.cursor()
    for division in divisions:
        division_id = get_division_id(division['_about'])
        title = division['title']
        try:
            curs.execute("INSERT INTO DIVISION (ID, DATE, TITLE) VALUES (?, ?, ?)",
                         (division_id, division_date, title))
            conn.commit()
        except sqlite3.OperationalError:
            print('FAILED DIVISION INSERT: {} - {} - {}'.format(division_id, division_date, title))
    conn.close()


def get_member_id(about):
    """ Returns the member id given the about field from the division json """
    return about[34:]


def get_member_vote(vote_type):
    """ Returns the member vote given the type field from the division json """
    return vote_type[38:]


def get_member_data(member_id, session):
    """ Returns the relevant data for an mp given their id """
    url = 'http://lda.data.parliament.uk/members/{}.json'.format(member_id)
    obj = session.get(url).json()
    primary_topic = obj['result']['primaryTopic']

    try:
        full_name = primary_topic['fullName']['_value']
    except KeyError:
        full_name = ''

    try:
        given_name = primary_topic['givenName']['_value']
    except KeyError:
        given_name = ''

    try:
        additional_name = primary_topic['additionalName']['_value']
    except KeyError:
        additional_name = ''

    try:
        family_name = primary_topic['familyName']['_value']
    except KeyError:
        family_name = ''

    try:
        party = primary_topic['party']['_value']
    except KeyError:
        party = ''

    try:
        constituency = primary_topic['constituency']['label']['_value']
    except KeyError:
        constituency = ''

    ret = {
        'full_name' : full_name,
        'given_name' : given_name,
        'additional_name' : additional_name,
        'family_name' : family_name,
        'party' : party,
        'constituency' : constituency,
    }

    return ret


def fill_member_and_vote_tables():
    """ Fills the Member and Vote tables in the database """
    conn = sqlite3.connect(DB_PATH)
    curs = conn.cursor()
    curs.execute("SELECT ID FROM DIVISION")
    rows = curs.fetchall()
    with requests.Session() as session:
        for row in rows:
            division_id = row[0]
            print(division_id)
            url = 'http://lda.data.parliament.uk/commonsdivisions/id/{}.json'.format(division_id)
            obj = session.get(url).json()
            votes = obj['result']['primaryTopic']['vote']
            for vote in votes:
                member_id = get_member_id(vote['member'][0]['_about'])
                member_vote = get_member_vote(vote['type'])
                curs.execute("SELECT count(*) FROM MEMBER WHERE ID = ?", (member_id,))
                data = curs.fetchone()[0]
                if data == 0:
                    member_data = get_member_data(member_id, session)
                    try:
                        curs.execute('''INSERT INTO MEMBER
                                        (ID, FULL_NAME, GIVEN_NAME, ADDITIONAL_NAME, FAMILY_NAME, PARTY, CONSTITUENCY)
                                        VALUES (?, ?, ?, ?, ?, ?, ?)''',
                                     (member_id, member_data['full_name'],
                                      member_data['given_name'], member_data['additional_name'],
                                      member_data['family_name'], member_data['party'],
                                      member_data['constituency']))
                        conn.commit()
                    except sqlite3.OperationalError:
                        print('FAILED MEMBER INSERT: {}'.format(member_data['full_name']))
                try:
                    curs.execute("INSERT INTO VOTE (MEMBER_ID, DIVISION_ID, VOTE) VALUES (?, ?, ?)",
                                 (member_id, division_id, member_vote))
                    conn.commit()
                except sqlite3.OperationalError:
                    print('FAILED VOTE INSERT: {} - {} - {}'
                          .format(member_id, division_id, member_vote))
                except sqlite3.IntegrityError:
                    print('FAILED VOTE INSERT (DUPLICATE): {} - {} - {}'
                          .format(member_id, division_id, member_vote))

    conn.close()


def run():
    """ Creates the database and fills it """
    create_tables()
    start_date = date(2001, 9, 11)
    end_date = date(2003, 3, 19)
    for day in date_range(start_date, end_date):
        division_inserts(day)

    fill_member_and_vote_tables()
    generate_divisions_csv()
    generate_members_csv()
    generate_votes_csv()


run()
