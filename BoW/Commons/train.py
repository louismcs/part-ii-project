import collections
import math
import re
import sqlite3

from nltk import ngrams
from nltk import PorterStemmer
from nltk.corpus import stopwords
from numpy import array_split
from numpy import linalg
from numpy import mean
from numpy import std
from random import shuffle
from scipy import stats
from sklearn import svm
from sklearn.metrics import accuracy_score
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


def get_members_from_file(file_path):
    """ Returns a list of ids of MPs to be reserved for testing given the file path """

    with open(file_path) as id_file:
        ret = [line.rstrip() for line in id_file]

    return ret

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


def get_aye_members_from_term(db_path, term, division_id):
    """ Returns a list of debate ids where the given term is in the debate title """

    conn = sqlite3.connect(db_path)
    curs = conn.cursor()

    curs.execute('''SELECT DISTINCT MEMBER_ID FROM SPEECH
                        WHERE DEBATE_URL IN (SELECT URL FROM DEBATE
                                             WHERE TITLE LIKE ? COLLATE NOCASE)
                        AND MEMBER_ID IN (SELECT ID FROM MEMBER INNER JOIN VOTE ON
                                          VOTE.MEMBER_ID = MEMBER.ID
                                          WHERE VOTE.DIVISION_ID=? AND
                                          (VOTE.VOTE='AyeVote')) ''',
                 ('%{}%'.format(term), division_id))

    rows = curs.fetchall()

    return [row[0] for row in rows]


def get_no_members_from_term(db_path, term, division_id):
    """ Returns a list of debate ids where the given term is in the debate title """

    conn = sqlite3.connect(db_path)
    curs = conn.cursor()

    curs.execute('''SELECT DISTINCT MEMBER_ID FROM SPEECH
                        WHERE DEBATE_URL IN (SELECT URL FROM DEBATE
                                             WHERE TITLE LIKE ? COLLATE NOCASE)
                        AND MEMBER_ID IN (SELECT ID FROM MEMBER INNER JOIN VOTE ON
                                          VOTE.MEMBER_ID = MEMBER.ID
                                          WHERE VOTE.DIVISION_ID=? AND
                                          (VOTE.VOTE='NoVote')) ''',
                 ('%{}%'.format(term), division_id))

    rows = curs.fetchall()

    return [row[0] for row in rows]


def get_all_member_ids(db_path, debate_terms, division_id):

    debates = set()
    for term in debate_terms:
        debates = debates.union(set(get_all_members_from_term(db_path, term, division_id)))

    return list(debates)


def get_aye_member_ids(db_path, debate_terms, division_id):
    
    debates = set()
    for term in debate_terms:
        debates = debates.union(set(get_aye_members_from_term(db_path, term, division_id)))

    return list(debates)


def get_no_member_ids(db_path, debate_terms, division_id):
    
    debates = set()
    for term in debate_terms:
        debates = debates.union(set(get_no_members_from_term(db_path, term, division_id)))

    return list(debates)


def get_mp_folds(settings):
    """ Given the number of folds, returns that number of
        non-overlapping lists (of equal/nearly equal length) of
        ids of mps matching the given settings """

    member_data = []
    for member in settings['training_mps']:
        votes = {}
        for division_id in settings['division_ids']:
            votes[division_id] = is_aye_vote(settings['db_path'], division_id, member)
        member_data.append({
            'id': member,
            'votes': votes
        })

    shuffle(member_data)

    test_folds = [list(element) for element in array_split(member_data, settings['no_of_folds'])]

    ret = []

    for test_fold in test_folds:
        train = [member for member in member_data if member not in test_fold]
        ret.append({
            'test': test_fold,
            'train': train
        })

    return ret


def get_speech_texts(db_path, member, debate):
    """ Returns a list of strings of the speeches of a given MP in a given debate """
    conn = sqlite3.connect(db_path)
    curs = conn.cursor()

    curs.execute('''SELECT QUOTE FROM SPEECH
                    WHERE MEMBER_ID=? AND DEBATE_URL=?''', (member['id'], debate))

    rows = curs.fetchall()

    return [{'text': row[0], 'votes': member['votes'], 'member': member['id']} for row in rows]


def get_speeches(db_path, member_list, debates):
    """ Returns all the speeches in the given database that match the given settings """
    speeches = {}
    for member in member_list:
        speeches[member['id']] = []
        for debate in debates:
            speeches[member['id']] = speeches[member['id']] + get_speech_texts(db_path, member, debate)

    return speeches

def fetch_speeches(speeches, mp_data):

    ret = []

    for member in mp_data:
        ret = ret + speeches[member['id']]

    return ret


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


def merge(n_gram):
    ret = ''
    for word in n_gram:
        ret += '{} '.format(word)

    return ret[:-1]
    
def get_n_grams(word_list, gram_size):
    """ Given a word list and some gram size, returns a list of all n grams for n <= gram_size """
    if gram_size == 1:
        ret = word_list
    else:
        ret = [merge(el) for el in ngrams(word_list, gram_size)] + get_n_grams(word_list, gram_size - 1)

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


def generate_question_bag(division_id, settings):

    ret = collections.Counter()

    with open('Data/motion{}.txt'.format(division_id), 'r') as motion_file:
        motion = motion_file.readlines()[0]

    word_list = generate_word_list(motion, settings)

    for word in word_list:
        ret[word] += 1

    return ret

def parse_ems(settings, mp_data, train):
    """ Parses a train ems file and creates the corresponding bags of words"""

    question_bags = {}

    for division_id in settings['division_ids']:
        question_bags[division_id] = generate_question_bag(division_id, settings)

    speeches = fetch_speeches(settings['speeches'], mp_data)

    aye_features = []
    no_features = []

    sum_bag = collections.Counter()

    members = {}

    for speech in speeches:
        if speech['member'] not in members:
            members[speech['member']] = speech['votes']
        word_list = generate_word_list(speech['text'], settings)
        bag = collections.Counter()
        for word in word_list:
            bag[word] += 1
            sum_bag[word] += 1
        if train:
            for division_id in settings['division_ids']:
                if speech['votes'][division_id]:
                    aye_features.append({
                        'speech_bag': bag,
                        'question_bag': question_bags[division_id],
                        'member': speech['member']
                    })
                else:
                    no_features.append({
                        'speech_bag': bag,
                        'question_bag': question_bags[division_id],
                        'member': speech['member']
                    })
        else:
            if speech['votes'][settings['test_division']]:
                aye_features.append({
                    'speech_bag': bag,
                    'question_bag': question_bags[settings['test_division']],
                    'member': speech['member']
                })
            else:
                no_features.append({
                    'speech_bag': bag,
                    'question_bag': question_bags[settings['test_division']],
                    'member': speech['member']
                })


    return aye_features, no_features, sum_bag, members


def condense_bags(features, words):
    """ Returns an array of integer arrays containing the counts of the words
         (in the array provided) and an array of the Counter bags """
    return [[feature['bag'][word] for word in words] for feature in features]


def normalise(feature):
    norm = linalg.norm(feature)
    if norm == 0:
        ret = feature
    else:
        ret = [el / norm for el in feature]

    return ret


def normalise_features(features):
    for feature in features:
        feature['bag'] = normalise(feature['bag'])

    return features

def generate_classifier_data(aye_features, no_features, common_words):
    """ Returns the features and samples in a form that can be used
         by a classifier, given the bags and most common words in them """

    features = aye_features + no_features
    features = condense_bags(features, common_words)
    features = normalise_features(features)
    samples = []

    for _ in range(len(aye_features)):
        samples.append(1)

    for _ in range(len(no_features)):
        samples.append(-1)

    return features, samples


def generate_train_data(settings, mp_list):
    """ Returns the features and samples in a form that can be used
         by a classifier, given the filenames for the data """

    aye_features, no_features, sum_bag, _ = parse_ems(settings, mp_list, True)
    if settings['max_bag_size']:
        common_words = [word[0] for word in sum_bag.most_common(settings['bag_size'])]
    else:
        common_words = []
        for term in sum_bag:
            if sum_bag[term] > 1:
                common_words.append(term)

    features, samples = generate_classifier_data(aye_features, no_features, common_words)

    return features, samples, common_words


def generate_test_data(common_words, settings, mp_list):
    """ Returns the features and samples in a form that can be used
         by a classifier, given the filenames for the data """

    aye_features, no_features, _, members = parse_ems(settings, mp_list, False)

    features, samples = generate_classifier_data(aye_features, no_features, common_words)

    return features, samples, members


def count_ayes(speeches):
    ret = 0
    for speech in speeches:
        if speech['prediction'] == 1:
            ret += 1

    return ret


def get_complete_bags(features):

    return [feature['speech_bag'] + feature['question_bag'] for feature in features]

def compute_f1(settings, data):
    """ Runs one loop of the cross-validation """

    train_features, train_samples, common_words = generate_train_data(settings, data['train'])

    test_features, test_samples, _ = generate_test_data(common_words, settings, data['test'])

    classifier = svm.SVC()
    ''' train_features is a list of word bags
        train_samples is a list containing only 1s and -1s
            (corresponding to the class ie an MP's vote) '''

    

    complete_train_features = get_complete_bags(train_features)
    complete_test_features = get_complete_bags(test_features)

    classifier.fit(complete_train_features, train_samples)
    test_predictions = classifier.predict(complete_test_features)

    return f1_score(test_samples, test_predictions)


def compute_member_f1s(settings, data):
    """ Runs one loop of the cross-validation """

    train_features, train_samples, common_words = generate_train_data(settings, data['train'])

    test_features, test_samples, members = generate_test_data(common_words, settings, data['test'])

    classifier = svm.SVC()
    ''' train_features is a list of word bags
        train_samples is a list containing only 1s and -1s
            (corresponding to the class ie an MP's vote) '''
    
    complete_train_features = get_complete_bags(train_features)
    complete_test_features = get_complete_bags(test_features)

    classifier.fit(complete_train_features, train_samples)

    test_predictions = classifier.predict(complete_test_features)

    grouped_speeches = {}

    for member_id in settings['testing_mps']:
        grouped_speeches[member_id] = {
            'votes': members[member_id],
            'speeches': []
        }

    for i, feature in enumerate(test_features):
        grouped_speeches[feature['member']['member_id']]['speeches'].append({
            'feature': complete_test_features[i],
            'prediction': test_predictions[i]
        })

    member_votes = []

    member_predictions = []

    for member_id in settings['testing_mps']:
        grouped_speeches[member_id]['aye_fraction'] = count_ayes(grouped_speeches[member_id]['speeches']) / len(grouped_speeches[member_id]['speeches'])
        grouped_speeches[member_id]['overall_prediction'] = 1 if grouped_speeches[member_id]['aye_fraction'] > 0.5 else -1
        member_votes.append(1 if grouped_speeches[member_id]['votes'][settings['test_division']] else -1)
        member_predictions.append(grouped_speeches[member_id]['overall_prediction'])

    #print(grouped_speeches)

    print('Accuracy by MP: {}%'.format(100 * accuracy_score(member_votes, member_predictions)))
    print('F1 by MP: {}'.format(f1_score(member_votes, member_predictions)))

    print('Accuracy by speech: {}%'.format(100 * accuracy_score(test_samples, test_predictions)))
    print('F1 by speech: {}'.format(f1_score(test_samples, test_predictions)))

    return f1_score(test_samples, test_predictions)


def compute_t(differences):
    mean = sum(differences) / len(differences)

    squared_diff = 0

    for difference in differences:
        squared_diff += pow(difference - mean, 2)

    variance = squared_diff / (len(differences) - 1)

    return mean / (math.sqrt(variance / len(differences)))


def change_n_gram(settings, increment, current_f1s, mp_folds):

    if settings['n_gram'] > 9 or settings['n_gram'] < 2:
        significant_change = False
    else:
        significant_change = True

    while significant_change:
        settings['n_gram'] += increment
        new_f1s = [compute_f1(settings, mp_fold) for mp_fold in mp_folds]

        current_mean = mean(current_f1s)
        new_mean = mean(new_f1s)

        print('New mean for n = {} is {}'.format(settings['n_gram'], new_mean))
        if new_mean > current_mean:
            current_f1s = new_f1s
            if settings['n_gram'] == 10 or settings['n_gram'] == 1:
                significant_change = False
        else:
            significant_change = False
            settings['n_gram'] -= increment

    return current_f1s


def choose_boolean_setting(settings, setting, current_f1s, mp_folds):

    settings[setting] = True
    new_f1s = [compute_f1(settings, mp_fold) for mp_fold in mp_folds]

    current_mean = mean(current_f1s)
    new_mean = mean(new_f1s)

    if new_mean > current_mean:
        current_f1s = new_f1s
    else:
        settings[setting] = False

    return current_f1s


def learn_settings(settings, mp_folds):

    current_f1s = [compute_f1(settings, mp_fold) for mp_fold in mp_folds]

    print(current_f1s)

    current_f1s = change_n_gram(settings, 1, current_f1s, mp_folds)

    print(current_f1s)

    current_f1s = choose_boolean_setting(settings, 'remove_stopwords', current_f1s, mp_folds)

    print(current_f1s)

    current_f1s = choose_boolean_setting(settings, 'stem_words', current_f1s, mp_folds)

    print(current_f1s)

    current_f1s = choose_boolean_setting(settings, 'group_numbers', current_f1s, mp_folds)

    print(current_f1s)

    current_mean = mean(current_f1s)

    if settings['n_gram'] > 1:
        settings['n_gram'] -= 1
        lower_n_f1s = [compute_f1(settings, mp_fold) for mp_fold in mp_folds]
        lower_n_mean = mean(lower_n_f1s)
    else:
        lower_n_mean = 0

    if settings['n_gram'] < 10:
        settings['n_gram'] += 2
        higher_n_f1s = [compute_f1(settings, mp_fold) for mp_fold in mp_folds]
        higher_n_mean = mean(higher_n_f1s)
    else:
        higher_n_mean = 0

    if higher_n_mean > lower_n_mean:
        if higher_n_mean > current_mean:
            current_f1s = change_n_gram(settings, 1, higher_n_f1s, mp_folds)
    else:
        if lower_n_mean > current_mean:
            settings['n_gram'] -= 2
            current_f1s = change_n_gram(settings, -1, lower_n_f1s, mp_folds)
    
    print('Average F1: {} ± {}'.format(mean(current_f1s), std(current_f1s)))


def is_aye_vote(db_path, division_id, member_id):

    conn = sqlite3.connect(db_path)
    curs = conn.cursor()

    curs.execute('''SELECT VOTE FROM VOTE WHERE MEMBER_ID=? AND DIVISION_ID=? ''',
                 (member_id, division_id))

    return curs.fetchone()[0] == 'AyeVote'


def run():
    """ Sets the settings and runs the program """

    settings = {
        'db_path': 'Data/Corpus/database.db',
        'black_list': [],
        'white_list': [],
        'bag_size': 100,
        'max_bag_size': True,
        'remove_stopwords': False,
        'stem_words': False,
        'group_numbers': False,
        'n_gram': 1,
        'test_division': 102564,
        'all_debates': False,
        'debate_terms': ['iraq', 'terrorism', 'middle east', 'defence policy',
                         'defence in the world', 'afghanistan'],
        'no_of_folds': 10,
        'entailment': True,
        'division_ids': [102564, 102565],
        'test_mp_file': 'test_data_combined.txt',
        'train_mp_file': 'train_data_combined.txt'
    }

    settings['debates'] = get_debates(settings)

    settings['testing_mps'] = get_members_from_file(settings['test_mp_file'])

    settings['training_mps'] = get_members_from_file(settings['train_mp_file'])

    mp_folds = get_mp_folds(settings)

    train_data = mp_folds[0]['test'] + mp_folds[0]['train']

    test_data = []

    for member in settings['testing_mps']:
        votes = {}
        for division_id in settings['division_ids']:
            votes[division_id] = is_aye_vote(settings['db_path'], division_id, member)
        test_data.append({
            'id': member,
            'votes': votes
        })

    member_data = train_data + test_data

    settings['speeches'] = get_speeches(settings['db_path'], member_data, settings['debates'])

    print('Got speeches')

    learn_settings(settings, mp_folds)

    print('N-gram: {}'.format(settings['n_gram']))
    print('Remove stopwords: {}'.format(settings['remove_stopwords']))
    print('Stem words: {}'.format(settings['stem_words']))
    print('Group numbers: {}'.format(settings['group_numbers']))
    print()


    ''' data = {
        'train': train_data,
        'test': test_data
    }

    compute_member_f1s(settings, data) '''

run()
