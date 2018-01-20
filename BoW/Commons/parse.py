""" Learns to guess how an MP votes, given transcriptions of House of Commons Sittings """

import collections
import re
import sqlite3

from nltk import PorterStemmer, ngrams
from nltk.corpus import stopwords
from numpy import array_split
from sklearn import svm
from sklearn.metrics import f1_score


def get_mps(settings, vote):
    """ Returns a list of mp ids who voted a given way in a given division """
    conn = sqlite3.connect(settings['db_path'])
    curs = conn.cursor()

    curs.execute('''SELECT ID FROM MEMBER
                    INNER JOIN VOTE ON VOTE.MEMBER_ID = MEMBER.ID
                    WHERE VOTE.DIVISION_ID=? AND VOTE.VOTE=?''', (settings['division_id'], vote))

    rows = curs.fetchall()

    return [row[0] for row in rows]


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

    #print('DEBATES: {}'.format(ret))

    return ret


def get_speech_texts(db_path, member_id, debate):
    """ Returns a list of strings of the speeches of a given MP in a given debate """
    conn = sqlite3.connect(db_path)
    curs = conn.cursor()

    curs.execute('''SELECT QUOTE FROM SPEECH
                    WHERE MEMBER_ID=? AND DEBATE_URL=?''', (member_id, debate))

    rows = curs.fetchall()

    return [row[0] for row in rows]

def get_all_speech_texts(db_path, mp_list, debates):
    """ Returns a list of strings of the speeches of given MPs in given debates """

    speeches = []

    for member_id in mp_list:
        #print('MEMBER DONE: {}'.format(member_id))
        for debate in debates:
            speeches = speeches + get_speech_texts(db_path, member_id, debate)

    return speeches

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

def parse_train_ems(settings):
    """ Parses a train ems file and creates the corresponding bags of words"""

    speeches = get_train_speeches(settings)

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

def generate_train_data(settings):
    """ Returns the features and samples in a form that can be used
         by a classifier, given the filenames for the data """

    aye_bags, no_bags, sum_bag = parse_train_ems(settings)

    common_words = [word[0] for word in sum_bag.most_common(settings['bag_size'])]

    #print("COMMON WORDS: {}".format(common_words[:20]))

    features, samples = generate_classifier_data(aye_bags, no_bags, common_words)

    return features, samples, common_words


def get_test_speeches(settings):
    """ Returns all the speeches in the given database that match the given settings """
    return get_speeches(settings, False)


def parse_test_ems(settings):
    """ Parses a test ems file and creates the corresponding bags of words"""

    speeches = get_test_speeches(settings)

    aye_bags = []
    no_bags = []

    for speech in speeches:
        word_list = generate_word_list(speech['text'], settings)
        bag = collections.Counter()
        for word in word_list:
            bag[word] += 1

        if speech['aye']:
            aye_bags.append(bag)
        else:
            no_bags.append(bag)

    return aye_bags, no_bags

def generate_test_data(common_words, settings):
    """ Returns the features and samples in a form that can be used
         by a classifier, given the filenames for the data """

    aye_bags, no_bags = parse_test_ems(settings)

    features, samples = generate_classifier_data(aye_bags, no_bags, common_words)

    return features, samples


def cross_validate(settings):
    """ Runs one loop of the cross-validation """


    train_features, train_samples, common_words = generate_train_data(settings)

    test_features, test_samples = generate_test_data(common_words, settings)

    classifier = svm.SVC()
    ''' train_features is a list of word bags
        train_samples is a list containing only 1s and -1s
            (corresponding to the class ie an MP's vote) '''
    classifier.fit(train_features, train_samples)

    test_predictions = classifier.predict(test_features)
    print('PREDICTIONS: {}'.format(test_predictions))

    print('NUM OF PREDICTIONS: {}'.format(len(test_predictions)))

    print('SCORE: {}'.format(classifier.score(test_features, test_samples)))

    print('F1 SCORE: {}'.format(f1_score(test_samples, test_predictions)))


def get_mp_folds(settings):
    """ Given the number of folds, returns that number of
        non-overlapping lists (of equal/nearly equal length) of
        ids of mps matching the given settings """

    all_mps = get_mps(settings, 'AyeVote') + get_mps(settings, 'NoVote')

    return [list(element) for element in array_split(all_mps, settings['no_of_folds'])]


def run():
    """ Sets the settings and runs the program """

    settings = {
        'db_path': 'Data/Corpus/database.db',
        'black_list': [],
        'white_list': [],
        'bag_size': 100,
        'remove_stopwords': True,
        'stem_words': False,
        'group_numbers': True,
        'n_gram': 1,
        'division_id': 102564,
        'all_debates': False,
        'debate_terms': ['iraq'],
        'no_of_folds': 10,
        'testing_mps': [],
    }

    mp_lists = get_mp_folds(settings)

    for mp_list in mp_lists:
        #print('TEST MPs: {}'.format(mp_list))
        settings['testing_mps'] = mp_list
        cross_validate(settings)


run()
