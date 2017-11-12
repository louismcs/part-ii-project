"""Creates, initialises and fills the database"""
import sqlite3
import csv

def setup_mp_table():
    """Sets up the table of MPs"""
    conn = sqlite3.connect('Data/Corpus/corpus5.db')
    conn.execute('''CREATE TABLE MP
            (ID         TEXT   PRIMARY KEY  NOT NULL,
             FIRSTNAME  TEXT                NOT NULL,
             SURNAME    TEXT                NOT NULL,
             PARTY      TEXT                NOT NULL,
             VOTE1      TEXT,
             VOTE2      TEXT);''')
    conn.close()

def add_mp_votes():
    """Appends the MP table to add their voting records on Iraq"""
    conn = sqlite3.connect('Data/Corpus/corpus5.db')
    curs = conn.cursor()
    with open('Data/VoteList.csv') as csvfile:
        vote_list = csv.reader(csvfile, delimiter=',')
        for row in vote_list:
            mp_id = row[0]
            vote_2 = row[1]
            vote_1 = row[2]
            curs.execute("UPDATE MP SET VOTE1 = ?, VOTE2 = ? WHERE ID = ?",
                         (vote_1, vote_2, mp_id))
            conn.commit()
    conn.close

def fill_mp_table():
    """Fills the MP table with the relevant data"""
    conn = sqlite3.connect('Data/Corpus/corpus5.db')
    curs = conn.cursor()
    with open('Data/MPList.csv') as csvfile:
        mp_list = csv.reader(csvfile, delimiter=',')
        for row in mp_list:
            mp_id = row[0]
            firstname = row[1]
            surname = row[2]
            party = row[3]
            try:
                curs.execute("INSERT INTO MP (ID,FIRSTNAME,SURNAME,PARTY,VOTE1,VOTE2) " \
                    "VALUES (?, ?, ?, ?, 6, 6)", (mp_id, firstname, surname, party))
                conn.commit()
            except sqlite3.OperationalError:
                print('FAILED: {} - {} - {} - {}'.format(mp_id, firstname, surname, party))
    conn.close()
    add_mp_votes()

def run():
    """Runs the program"""
    ''' setup_mp_table()
    fill_mp_table() '''
    conn = sqlite3.connect('Data/Corpus/corpus5.db')
    cursor = conn.execute("SELECT SURNAME,PARTY,VOTE1,VOTE2 from MP")
    count = 0
    for row in cursor:
        print('{}: {} - {} - {} - {}'.format(count, row[0], row[1], row[2], row[3]))
        count += 1

    ''' Check if it stops after the same number of columns '''

run()
