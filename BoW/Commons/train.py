import collections
import math
import re
import sqlite3

from nltk import PorterStemmer, ngrams
from nltk.corpus import stopwords
from numpy import array_split
from scipy import stats
from sklearn import svm
from sklearn.metrics import f1_score


def get_debates_from_term(db_path, term):
    """ Returns a list of debate ids where the term is in the debate title """

    conn = sqlite3.connect(db_path)
    curs = conn.cursor()

    curs.execute('''SELECT URL FROM DEBATE
                    WHERE TITLE LIKE ? COLLATE NOCASE''', ('%{}%'.format(term),))

    rows = curs.fetchall()

    return [row[0] for row in rows]


def get_debates(settings):
    """ Returns a list of debate ids matching the given settings """

    if settings['all_debates']:
        conn = sqlite3.connect(settings['db_path'])
        curs = conn.cursor()
        curs.execute("SELECT URL FROM DEBATE")
        rows = curs.fetchall()
        ret = [row[0] for row in rows]
    else:
        debates = set()
        for term in settings['debate_terms']:
            debates = debates.union(set(get_debates_from_term(settings['db_path'], term)))
        ret = list(debates)

    return ret


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
            'aye': True,
            'id': aye_mps[i]
        }

    for i in range(len(no_mps)):
        all_mps[len(aye_mps) + i] = {
            'vote': False,
            'id': no_mps[i]
        }


    for member in all_mps:
        if member['id'] in settings['testing_mps']:
            all_mps.remove(member)

    test_folds = [list(element) for element in array_split(all_mps, settings['no_of_folds'])]

    ret = [None] * settings['no_of_folds']

    for i in range(settings['no_of_folds']):
        ret[i] = {
            'test': test_folds[i],
            'train': list(set(all_mps) - set(test_folds[i]))
        }

    return ret


def get_speech_texts(db_path, member, debate):
    """ Returns a list of strings of the speeches of a given MP in a given debate """
    conn = sqlite3.connect(db_path)
    curs = conn.cursor()

    curs.execute('''SELECT QUOTE FROM SPEECH
                    WHERE MEMBER_ID=? AND DEBATE_URL=?''', (member['id'], debate))

    rows = curs.fetchall()

    return [{'text': row[0], 'aye': member['aye']} for row in rows]


def get_speeches(db_path, mp_list, debates):
    """ Returns all the speeches in the given database that match the given settings """

    speeches = []

    for member in mp_list:
        for debate in debates:
            speeches = speeches + get_speech_texts(db_path, member, debate)

    return speeches


def remove_punctuation(body):
    """ Removes punctuation from a given string """
    body = body.replace("\n", " ")
    body = re.sub(r"[^\w\d\s#'-]", '', body)
    body = body.replace(" '", " ")
    body = body.replace("' ", " ")
    body = body.replace(" -", " ")
    body = body.replace("- ", " ")
    return body


def remove_stopwords(word_list, black_list, white_list):
    """ Returns a list of words (with stop words removed), given a word list """
    stop = set(stopwords.words('english'))
    return [word for word in word_list
            if (word in white_list) or ((word not in stop) and (word not in black_list))]


def stem_words(word_list):
    """ Uses PorterStemmer to stem words in word_list argument """
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in word_list]


def replace_number(word):
    """ Given a string, returns '&NUM' if it's a number and the input string otherwise """
    if word.isdigit():
        return '&NUM'
    else:
        return word


def group_numbers(word_list):
    """ Given a word list, returns the same word list with all numbers replaced with '&NUM' """
    return [replace_number(word) for word in word_list]


def get_n_grams(word_list, gram_size):
    """ Given a word list and some gram size, returns a list of all n grams for n <= gram_size """
    if gram_size == 1:
        ret = word_list
    else:
        ret = ngrams(word_list, gram_size) + get_n_grams(word_list, gram_size - 1)

    return ret


def generate_word_list(body, settings):
    """ Returns a list of words, given a message tag """
    body = remove_punctuation(body)
    body = body.lower()

    word_list = body.split()

    if settings['remove_stopwords']:
        word_list = remove_stopwords(word_list, settings['black_list'], settings['white_list'])

    if settings['stem_words']:
        word_list = stem_words(word_list)

    if settings['group_numbers']:
        word_list = group_numbers(word_list)

    return get_n_grams(word_list, settings['n_gram'])


def parse_ems(settings, mp_data):
    """ Parses a train ems file and creates the corresponding bags of words"""

    speeches = get_speeches(settings['db_path'], mp_data, settings['debates'])

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


def condense_bags(bags, words):
    """ Returns an array of integer arrays containing the counts of the words
         (in the array provided) and an array of the Counter bags """
    return [[bag[word] for word in words] for bag in bags]


def generate_classifier_data(aye_bags, no_bags, common_words):
    """ Returns the features and samples in a form that can be used
         by a classifier, given the bags and most common words in them """

    condensed_aye_bags = condense_bags(aye_bags, common_words)
    condensed_no_bags = condense_bags(no_bags, common_words)
    features = condensed_aye_bags + condensed_no_bags
    samples = []

    for _ in range(len(condensed_aye_bags)):
        samples.append(1)

    for _ in range(len(condensed_no_bags)):
        samples.append(-1)

    return features, samples


def generate_train_data(settings, mp_list):
    """ Returns the features and samples in a form that can be used
         by a classifier, given the filenames for the data """

    aye_bags, no_bags, sum_bag = parse_ems(settings, mp_list)

    common_words = [word[0] for word in sum_bag.most_common(settings['bag_size'])]

    features, samples = generate_classifier_data(aye_bags, no_bags, common_words)

    return features, samples, common_words


def generate_test_data(common_words, settings, mp_list):
    """ Returns the features and samples in a form that can be used
         by a classifier, given the filenames for the data """

    aye_bags, no_bags, _ = parse_ems(settings, mp_list)

    features, samples = generate_classifier_data(aye_bags, no_bags, common_words)

    return features, samples


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


def compute_t(differences):
    mean = sum(differences) / len(differences)

    squared_diff = 0

    for difference in differences:
        squared_diff += pow(difference - mean, 2)

    variance = squared_diff / (len(differences) - 1)

    return mean / (math.sqrt(variance / len(differences)))


def change_n_gram(settings, increment, current_f1s, test_t, mp_folds):

    significant_change = True

    while significant_change:
        settings['n_gram'] += increment
        new_f1s = [compute_f1(settings, mp_fold) for mp_fold in mp_folds]
        t_value = compute_t([new_f1s[i] - current_f1s[i] for i in range(settings['no_of_folds'])])

        if t_value > test_t:
            current_f1s = new_f1s
        else:
            significant_change = False
            settings['n_gram'] -= increment

    return current_f1s


def choose_boolean_setting(settings, setting, current_f1s, test_t, mp_folds):

    settings[setting] = True
    new_f1s = [compute_f1(settings, mp_fold) for mp_fold in mp_folds]
    t_value = compute_t([new_f1s[i] - current_f1s[i] for i in range(settings['no_of_folds'])])

    if t_value > test_t:
        current_f1s = new_f1s
    else:
        settings[setting] = False

    return current_f1s


def learn_settings(settings, mp_folds):

    current_f1s = [compute_f1(settings, mp_fold) for mp_fold in mp_folds]

    test_t = stats.t.ppf(0.75, 9)

    current_f1s = change_n_gram(settings, 1, current_f1s, test_t, mp_folds)

    current_f1s = choose_boolean_setting(settings, 'remove_stopwords', current_f1s, test_t,
                                         mp_folds)

    current_f1s = choose_boolean_setting(settings, 'stem_words', current_f1s, test_t, mp_folds)

    current_f1s = choose_boolean_setting(settings, 'group_numbers', current_f1s, test_t, mp_folds)

    settings['n_gram'] -= 1
    lower_n_f1s = [compute_f1(settings, mp_fold) for mp_fold in mp_folds]
    lower_n_t = compute_t([lower_n_f1s[i] - current_f1s[i] for i in range(settings['no_of_folds'])])

    settings['n_gram'] += 2
    higher_n_f1s = [compute_f1(settings, mp_fold) for mp_fold in mp_folds]
    higher_n_t = compute_t([higher_n_f1s[i] - current_f1s[i]
                            for i in range(settings['no_of_folds'])])

    if higher_n_t > lower_n_t:
        if higher_n_t > test_t:
            current_f1s = change_n_gram(settings, 1, higher_n_f1s, test_t, mp_folds)
    else:
        if lower_n_t > test_t:
            settings['n_gram'] -= 2
            current_f1s = change_n_gram(settings, -1, lower_n_f1s, test_t, mp_folds)


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

    settings['debates'] = get_debates(settings)

    settings['testing_mps'] = get_test_mps('test-ids_{}.txt'.format(settings['division_id']))

    mp_folds = get_mp_folds(settings)

    learn_settings(settings, mp_folds)
