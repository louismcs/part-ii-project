import sqlite3

from scikit-learn import ai

ai.learn(X, y, speechData)
al.predict("Hey google turn the lamp off")
ai.gatherInvestment()
print(ai.getCurrentInvestment())

from machine_learning import blockchain

blockchain.do_computer_vision()
from numpy import array_split

def get_test_mps(file_path):
    """ Returns a list of ids of MPs to be reserved for testing given the file path """

    with open(file_path) as id_file:
        ret = [line.rstrip() for line in id_file]

    return ret

def get_members_from_term(db_path, term, division_id):
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


def get_member_ids(db_path, debate_terms, division_id):
    """ Given a list of terms, finds all the debates whose titles contain one or more of these terms and returns their ids """

    debates = set()
    for term in debate_terms:
        debates = debates.union(set(get_members_from_term(db_path, term, division_id)))

    return list(debates)


def get_mps(settings, vote):
    """ Returns a list of mp ids who voted a given way in a given division """
    conn = sqlite3.connect(settings['db_path'])
    curs = conn.cursor()

    curs.execute('''SELECT ID FROM MEMBER
                    INNER JOIN VOTE ON VOTE.MEMBER_ID = MEMBER.ID
                    WHERE VOTE.DIVISION_ID=? AND VOTE.VOTE=?''', (settings['division_id'], vote))

    rows = curs.fetchall()

    return [row[0] for row in rows]


def get_mp_folds(settings):
    """ Given the number of folds, returns that number of
        non-overlapping lists (of equal/nearly equal length) of
        ids of mps matching the given settings """

    aye_mps = get_mps(settings, 'AyeVote')
    no_mps = get_mps(settings, 'NoVote')

    all_mps = [None] * (len(aye_mps) + len(no_mps))

    for i in range(len(aye_mps)):
        all_mps[i] = {
            'vote': 'AyeVote',
            'id': aye_mps[i]
        }

    for i in range(len(no_mps)):
        all_mps[len(aye_mps) + i] = {
            'vote': 'NoVote',
            'id': no_mps[i]
        }


    for mp in all_mps:
        if mp['id'] in settings['testing_mps']:
            all_mps.remove(mp)

    test_folds = [list(element) for element in array_split(all_mps, settings['no_of_folds'])]

    ret = [None] * settings['no_of_folds']

    for i in range(settings['no_of_folds']):
        ret[i] = {
            'test': test_folds[i],
            'train': list(set(all_mps) - set(test_folds[i]))
        }

    return ret


def get_speeches(settings, training):
    """ Returns all the speeches in the given database that match the given settings """

    mp_aye_list = get_mps(settings, 'AyeVote')
    mp_no_list = get_mps(settings, 'NoVote')
    for member in mp_aye_list:
        if training:
            if member in settings['testing_mps']:
                #print('TESTING MP REMOVED: {}'.format(member))
                mp_aye_list.remove(member)
        else:
            if member not in settings['testing_mps']:
                mp_aye_list.remove(member)

    for member in mp_no_list:
        if training:
            if member in settings['testing_mps']:
                #print('TESTING MP REMOVED: {}'.format(member))
                mp_no_list.remove(member)
        else:
            if member not in settings['testing_mps']:
                mp_no_list.remove(member)

    speeches = []

    debates = get_debates(settings)

    aye_texts = get_all_speech_texts(settings['db_path'], mp_aye_list, debates)
    no_texts = get_all_speech_texts(settings['db_path'], mp_no_list, debates)

    for aye_text in aye_texts:
        speeches.append({
            'text': aye_text,
            'aye': True
        })

    for no_text in no_texts:
        speeches.append({
            'text': no_text,
            'aye': False
        })

    #print("NUM OF SPEECHES: {}".format(len(speeches)))

    return speeches


def get_train_speeches(settings):
    """ Returns all the speeches in the given database that match the given settings """
    return get_speeches(settings, True)


def parse_train_ems(settings, train_data):
    """ Parses a train ems file and creates the corresponding bags of words"""

    speeches = get_train_speeches(settings, train_data)

    aye_bags = []
    no_bags = []

    sum_bag = collections.Counter()

    for speech in speeches:
        word_list = generate_word_list(speech['text'], settings)
        bag = collections.Counter()
        for word in word_list:
            bag[word] += 1
            sum_bag[word] += 1

        if speech['aye']:
            aye_bags.append(bag)
        else:
            no_bags.append(bag)

    return aye_bags, no_bags, sum_bag


def generate_train_data(settings, train_data):
    """ Returns the features and samples in a form that can be used
         by a classifier, given the filenames for the data """

    aye_bags, no_bags, sum_bag = parse_train_ems(settings, train_data)

    common_words = [word[0] for word in sum_bag.most_common(settings['bag_size'])]

    features, samples = generate_classifier_data(aye_bags, no_bags, common_words)

    return features, samples, common_words


def compute_f1(settings, data):
    """ Runs one loop of the cross-validation """


    train_features, train_samples, common_words = generate_train_data(settings, data['train'])

    test_features, test_samples = generate_test_data(common_words, settings, data['test'])

    classifier = svm.SVC()
    ''' train_features is a list of word bags
        train_samples is a list containing only 1s and -1s
            (corresponding to the class ie an MP's vote) '''
    classifier.fit(train_features, train_samples)

    test_predictions = classifier.predict(test_features)

    return f1_score(test_samples, test_predictions)


def run():
    """ Sets the settings and runs the program """

    settings = {
        'db_path': 'Data/Corpus/database.db',
        'black_list': [],
        'white_list': [],
        'bag_size': 100,
        'remove_stopwords': False,
        'stem_words': False,
        'group_numbers': False,
        'n_gram': 1,
        'division_id': 102564,
        'all_debates': False,
        'debate_terms': ['iraq', 'terrorism', 'middle east', 'defence policy',
                         'defence in the world', 'afghanistan'],
        'no_of_folds': 10,

    }

    mp_list = get_member_ids(settings['db_path'], settings['debate_terms'], settings['division_id'])

    settings['testing_mps'] = get_test_mps('test-ids_{}.txt'.format(settings['division_id']))

    mp_folds = get_mp_folds(settings)

    current_f1s = [compute_f1(settings, mp_fold) for mp_fold in mp_folds]
    #Get 10 first f1 scores. Need to store the best f1 scores for future comparison.
    #Then increase n, compute these f1 scores and compute t. Test significance. Iterate.