""" Splits the data into training and test data """

import sqlite3
from random import shuffle


def get_members_from_term(db_path, term, division_id):
    """ Returns a list of debate ids where the term is in the debate title """

    conn = sqlite3.connect(db_path)
    curs = conn.cursor()

    curs.execute('''SELECT DISTINCT MEMBER_ID FROM SPEECH
                        WHERE DEBATE_URL IN (SELECT URL FROM DEBATE
                                             WHERE TITLE LIKE ? COLLATE NOCASE)
                        AND MEMBER_ID IN (SELECT ID FROM MEMBER INNER JOIN VOTE ON
                                          VOTE.MEMBER_ID = MEMBER.ID
                                          WHERE VOTE.DIVISION_ID=? AND
                                          (VOTE.VOTE='AyeVote' OR VOTE.VOTE='NoVote')) ''',
                 ('%{}%'.format(term), division_id))

    rows = curs.fetchall()

    return [row[0] for row in rows]


def get_member_ids(db_path, debate_terms, division_id):
    """ Given a list of terms, finds all the debates whose titles contain one or more of these terms and returns their ids """

    debates = set()
    for term in debate_terms:
        debates = debates.union(set(get_members_from_term(db_path, term, division_id)))

    return list(debates)


def is_aye_vote(db_path, division_id, member_id):

    conn = sqlite3.connect(db_path)
    curs = conn.cursor()

    curs.execute('''SELECT VOTE FROM VOTE WHERE MEMBER_ID=? AND DIVISION_ID=? ''',
                 (member_id, division_id))

    return curs.fetchone()[0] == 'AyeVote'


def get_debates_from_term(db_path, term):
    """ Returns a list of debate ids where the term is in the debate title """

    conn = sqlite3.connect(db_path)
    curs = conn.cursor()

    curs.execute('''SELECT URL FROM DEBATE
                    WHERE TITLE LIKE ? COLLATE NOCASE''', ('%{}%'.format(term),))

    rows = curs.fetchall()

    return [row[0] for row in rows]


def get_debate_ids(db_path, debate_terms):
    """ Returns a list of debate ids matching the given settings """

    debates = set()
    for term in debate_terms:
        debates = debates.union(set(get_debates_from_term(db_path, term)))
    return list(debates)


def get_number_of_speeches(db_path, debate_ids, member_ids):

    conn = sqlite3.connect(db_path)
    curs = conn.cursor()

    statement = ''' SELECT COUNT(*) FROM SPEECH WHERE DEBATE_URL IN ({debates})
                    AND MEMBER_ID IN ({members}) '''.format(
                        debates=','.join(['?']*len(debate_ids)),
                        members=','.join(['?']*len(member_ids)))

    curs.execute(statement, debate_ids + member_ids)

    return curs.fetchone()[0]

def choose_test_data(db_path, debate_terms, division_id):
    member_ids = get_member_ids(db_path, debate_terms, division_id)
    test_size = round(0.1 * len(member_ids))
    shuffle(member_ids)
    test_ids = member_ids[:test_size]
    train_ids = [member_id for member_id in member_ids if member_id not in test_ids]

    test_ayes = 0
    for test_id in test_ids:
        if is_aye_vote(db_path, division_id, test_id):
            test_ayes += 1

    test_percent = 100 * test_ayes / len(test_ids)

    train_ayes = 0
    for train_id in train_ids:
        if is_aye_vote(db_path, division_id, train_id):
            train_ayes += 1

    train_percent = 100 * train_ayes / len(train_ids)

    debate_ids = get_debate_ids(db_path, debate_terms)

    test_speeches = get_number_of_speeches(db_path, debate_ids, test_ids)

    train_speeches = get_number_of_speeches(db_path, debate_ids, train_ids)

    print('TEST AYES: {}\nTEST TOTAL: {}\nTEST AYES PERCENT: {}%\nTEST SPEECHES: {}\nTEST SPEECHES PER MP: {}\n'
          .format(test_ayes, len(test_ids), test_percent, test_speeches, test_speeches / len(test_ids)))
    print('TRAIN AYES: {}\nTRAIN TOTAL: {}\nTRAIN AYES PERCENT: {}%\nTRAIN SPEECHES: {}\nTRAIN SPEECHES PER MP: {}'
          .format(train_ayes, len(train_ids), train_percent, train_speeches, train_speeches / len(train_ids)))

    txt_file = open('testdata2.txt', 'w')

    for test_id in test_ids:
        txt_file.write('{}\n'.format(test_id))


def run():
    db_path = 'Data/Corpus/database.db'
    terms = ['iraq', 'terrorism', 'middle east',
             'defence policy', 'defence in the world', 'afghanistan']
    division_id = 102564
    choose_test_data(db_path, terms, division_id)

run()
