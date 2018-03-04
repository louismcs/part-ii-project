import sqlite3

from scipy.stats import spearmanr


def get_all_members_from_term(db_path, term, division_id):
    """ Returns a list of debate ids where the given term is in the debate title """

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


def get_all_member_ids(db_path, debate_terms, division_id):

    debates = set()
    for term in debate_terms:
        debates = debates.union(set(get_all_members_from_term(db_path, term, division_id)))

    return debates


def is_aye_vote(db_path, division_id, member_id):

    conn = sqlite3.connect(db_path)
    curs = conn.cursor()

    curs.execute('''SELECT VOTE FROM VOTE WHERE MEMBER_ID=? AND DIVISION_ID=? ''',
                 (member_id, division_id))

    return 1 if curs.fetchone()[0] == 'AyeVote' else 0


def run():
    debate_terms = ['iraq', 'terrorism', 'middle east', 'defence policy',
                         'defence in the world', 'afghanistan']
    db_path = 'Data/Corpus/database.db'

    vote_one_members = get_all_member_ids(db_path, debate_terms, 102564)
    vote_two_members = get_all_member_ids(db_path, debate_terms, 102565)
    
    member_ids = list(vote_one_members.intersection(vote_two_members))

    vote_ones = []
    vote_twos = []
    for member_id in member_ids:
        vote_ones.append(is_aye_vote(db_path, 102564, member_id))
        vote_twos.append(is_aye_vote(db_path, 102565, member_id))
    
    vote_one_ayes = sum(vote_ones)
    vote_one_nos = len(vote_ones) - vote_one_ayes

    
    vote_two_ayes = sum(vote_twos)
    vote_two_nos = len(vote_twos) - vote_two_ayes

    spearman_coefficient = spearmanr(vote_ones, vote_twos)

    print('Spearman coefficient: {}'.format(spearman_coefficient))

run()
