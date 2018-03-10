import collections
import math
import pickle
import re
import sqlite3

from nltk import ngrams
from nltk import PorterStemmer
from nltk.corpus import stopwords
from numpy import array
from numpy import array_split
from numpy import linalg
from numpy import log10
from numpy import logspace
from numpy import matmul
from numpy import mean
from numpy import sqrt
from numpy import std
from numpy.linalg import svd
from random import shuffle
from sklearn import svm
from sklearn.decomposition import SparsePCA
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

def get_member_no_of_speeches(db_path, debate_ids, member_id):
    conn = sqlite3.connect(db_path)
    curs = conn.cursor()

    statement = ''' SELECT COUNT(*) FROM SPEECH WHERE DEBATE_URL IN ({debates})
                    AND MEMBER_ID={member} '''.format(
                        debates=','.join(['?']*len(debate_ids)),
                        member=member_id)

    curs.execute(statement, debate_ids)

    return curs.fetchone()[0]


def get_mp_folds(settings):
    """ Given the number of folds, returns that number of
        non-overlapping lists (of equal/nearly equal length) of
        ids of mps matching the given settings """
    speech_counts = {}
    votes = {}
    speech_count_list = []
    total_ayes = {}
    for division_id in settings['division_ids']:
        total_ayes[division_id] = 0

    for member_id in settings['training_mps']:
        no_of_speeches = get_member_no_of_speeches(settings['db_path'], settings['debates'], member_id)
        speech_counts[member_id] = no_of_speeches
        speech_count_list.append(no_of_speeches)
        votes[member_id] = {}
        for division_id in settings['division_ids']:
            vote = is_aye_vote(settings['db_path'], division_id, member_id)
            votes[member_id][division_id] = vote
            if vote:
                total_ayes[division_id] += 1
    
    total_percents = {}
    for division_id in settings['division_ids']:
        total_percents[division_id] = 100 * total_ayes[division_id] / len(settings['training_mps'])

    mean_total_speeches = mean(speech_count_list)
    std_total_speeches = 4 * std(speech_count_list) / sqrt(len(speech_count_list))
    aye_percents_in_range = False
    no_of_speeches_in_range = False
    member_data = []
    for member in settings['training_mps']:
        member_data.append({
            'id': member,
            'votes': votes[member]
        })
    while not (aye_percents_in_range and no_of_speeches_in_range):
        shuffle(member_data)

        test_folds = [list(element) for element in array_split(member_data,
                                                               settings['no_of_folds'])]

        aye_percents_in_range = True
        no_of_speeches_in_range = True
        for fold in test_folds:
            fold_speech_count = 0

            for member in fold:
                fold_speech_count += speech_counts[member['id']]

            mean_fold_speech_count = fold_speech_count / len(fold)
            no_of_speeches_in_range = (no_of_speeches_in_range
                                       and mean_fold_speech_count >
                                       mean_total_speeches - std_total_speeches
                                       and mean_fold_speech_count <
                                       mean_total_speeches + std_total_speeches)

            for division_id in settings['division_ids']:
                aye_votes = 0
                for member in fold:
                    if member['votes'][division_id]:
                        aye_votes += 1

                aye_percent = 100 * aye_votes / len(fold)
                aye_percents_in_range = (aye_percents_in_range
                                         and aye_percent > total_percents[division_id] - 15
                                         and aye_percent < total_percents[division_id] + 15)

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


def normalise(feature):
    norm = linalg.norm(feature)
    if norm == 0:
        ret = feature
    else:
        ret = [el / norm for el in feature]

    return ret


def normalise_features(features):
    for feature in features:
        feature['speech_bag'] = normalise(feature['speech_bag'])

    return features


def generate_question_bags(settings):

    sum_bag = collections.Counter()
    bags = {}
    for division_id in settings['division_ids']:
        with open('Data/motion{}.txt'.format(division_id), 'r') as motion_file:
            motion = motion_file.readlines()[0]
        word_list = generate_word_list(motion, settings)
        bag = collections.Counter()
        for word in word_list:
            sum_bag[word] += 1
            bag[word] += 1
        bags[division_id] = bag

    ret = {}
    for division_id in settings['division_ids']:
        if settings['normalise']:
            ret[division_id] = normalise([bags[division_id][word] for word in sum_bag])
        else:
            ret[division_id] = [bags[division_id][word] for word in sum_bag]

    return ret


def parse_ems(settings, mp_data, train):
    """ Parses a train ems file and creates the corresponding bags of words"""
    if settings['entailment']:
        question_bags = generate_question_bags(settings)

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
        if ['entailment']:
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
        else:
            if speech['votes'][settings['test_division']]:
                aye_features.append({
                    'speech_bag': bag,
                    'member': speech['member']
                })
            else:
                no_features.append({
                    'speech_bag': bag,
                    'member': speech['member']
                })


    return aye_features, no_features, sum_bag, members


def condense_bag(feature, words):
    return [feature[word] for word in words]


def condense_bags(features, words):
    """ Returns an array of integer arrays containing the counts of the words
         (in the array provided) and an array of the Counter bags """

    for feature in features:
        feature['speech_bag'] = condense_bag(feature['speech_bag'], words)

    return features


def generate_classifier_data(aye_features, no_features, common_words, normalise_data):
    """ Returns the features and samples in a form that can be used
         by a classifier, given the bags and most common words in them """

    features = aye_features + no_features
    features = condense_bags(features, common_words)
    if normalise_data:
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
            if sum_bag[term] > 3:
                common_words.append(term)

    features, samples = generate_classifier_data(aye_features, no_features, common_words, settings['normalise'])

    return features, samples, common_words


def generate_test_data(common_words, settings, mp_list):
    """ Returns the features and samples in a form that can be used
         by a classifier, given the filenames for the data """

    aye_features, no_features, _, members = parse_ems(settings, mp_list, False)

    features, samples = generate_classifier_data(aye_features, no_features, common_words, settings['normalise'])

    return features, samples, members


def count_ayes(speeches):
    ret = 0
    for speech in speeches:
        if speech['prediction'] == 1:
            ret += 1

    return ret


def get_complete_bags(features, entailment):
    if entailment:
        ret = [feature['speech_bag'] + feature['question_bag'] for feature in features]
    else:
        ret = [feature['speech_bag'] for feature in features]
    
    return ret


def compute_rank(s):
    min_val = s[0] / 100

    i = 0
    val = s[i]
    while val > min_val:
        i += 1
        val = s[i]

    return i - 1


def compute_linear_fold_f1s(settings, data, linear_param_values):
    train_features, train_samples, common_words = generate_train_data(settings, data['train'])

    test_features, test_samples, _ = generate_test_data(common_words, settings, data['test'])

    complete_train_features = array(get_complete_bags(train_features, settings['entailment']))

    complete_test_features = array(get_complete_bags(test_features, settings['entailment']))

    ret = []

    for c_value in linear_param_values['cs']:
        classifier = svm.SVC(C=c_value, kernel='linear', cache_size=settings['cache'])

        classifier.fit(complete_train_features, train_samples)

        test_predictions = classifier.predict(complete_test_features)

        ret.append({
            'linear_params': {
                'c': c_value,
            },
            'f1': f1_score(test_samples, test_predictions)
        })

    return ret


def compute_rbf_fold_f1s(settings, data, rbf_param_values):
    train_features, train_samples, common_words = generate_train_data(settings, data['train'])

    test_features, test_samples, _ = generate_test_data(common_words, settings, data['test'])

    complete_train_features = array(get_complete_bags(train_features, settings['entailment']))

    complete_test_features = array(get_complete_bags(test_features, settings['entailment']))

    ret = []

    for c_value in rbf_param_values['cs']:
        for gamma_value in rbf_param_values['gammas']:
            classifier = svm.SVC(C=c_value, kernel='rbf', gamma=gamma_value, cache_size=settings['cache'])

            classifier.fit(complete_train_features, train_samples)

            test_predictions = classifier.predict(complete_test_features)

            ret.append({
                'rbf_params': {
                    'c': c_value,
                    'gamma': gamma_value
                },
                'f1': f1_score(test_samples, test_predictions)
            })

    return ret


def compute_poly_fold_f1s(settings, data, poly_param_values):
    train_features, train_samples, common_words = generate_train_data(settings, data['train'])

    test_features, test_samples, _ = generate_test_data(common_words, settings, data['test'])

    complete_train_features = array(get_complete_bags(train_features, settings['entailment']))

    complete_test_features = array(get_complete_bags(test_features, settings['entailment']))

    ret = []

    for c_value in poly_param_values['cs']:
        for gamma_value in poly_param_values['gammas']:
            for d_value in poly_param_values['ds']:
                for r_value in poly_param_values['rs']:
                    classifier = svm.SVC(C=c_value, kernel='poly', degree=d_value, gamma=gamma_value, coef0=r_value, cache_size=settings['cache'])

                    classifier.fit(complete_train_features, train_samples)

                    test_predictions = classifier.predict(complete_test_features)

                    ret.append({
                        'poly_params': {
                            'c': c_value,
                            'gamma': gamma_value,
                            'd': d_value,
                            'r': r_value
                        },
                        'f1': f1_score(test_samples, test_predictions)
                    })

    return ret


def compute_f1(settings, data):
    """ Runs one loop of the cross-validation """

    train_features, train_samples, common_words = generate_train_data(settings, data['train'])

    test_features, test_samples, _ = generate_test_data(common_words, settings, data['test'])

    if settings['kernel'] == 'linear':
        classifier = svm.SVC(C=settings['linear_c'], kernel='linear', cache_size=settings['cache'])
    elif settings['kernel'] == 'rbf':
        classifier = svm.SVC(C=settings['rbf_c'], kernel='rbf', gamma=settings['rbf_gamma'], cache_size=settings['cache'])
    else:
        #Assert kernel is poly
        classifier = svm.SVC(C=settings['poly_c'], kernel='poly', degree=settings['poly_d'], gamma=settings['poly_gamma'], coef0=settings['poly_r'], cache_size=settings['cache'])

    ''' train_features is a list of word bags
        train_samples is a list containing only 1s and -1s
            (corresponding to the class ie an MP's vote) '''

    

    complete_train_features = array(get_complete_bags(train_features, settings['entailment']))
    complete_test_features = array(get_complete_bags(test_features, settings['entailment']))

    '''

    print('features saved')

    _, sigma, v_transpose = svd(complete_train_features, full_matrices=True, compute_uv=True)

    rank = compute_rank(sigma)

    print('Rank: {}'.format(rank))

    truncated_v = v_transpose[:rank].transpose()

    reduced_train_features = matmul(complete_train_features, truncated_v)

    reduced_test_features = matmul(complete_test_features, truncated_v)

    print('Computed reduced matrices') '''

    classifier.fit(complete_train_features, train_samples)
    test_predictions = classifier.predict(complete_test_features)

    return f1_score(test_samples, test_predictions)


def compute_member_f1s(settings, data):
    """ Runs one loop of the cross-validation """

    train_features, train_samples, common_words = generate_train_data(settings, data['train'])

    test_features, test_samples, members = generate_test_data(common_words, settings, data['test'])

    if settings['kernel'] == 'linear':
        classifier = svm.SVC(C=settings['linear_c'], kernel='linear', cache_size=settings['cache'])
    elif settings['kernel'] == 'rbf':
        classifier = svm.SVC(C=settings['rbf_c'], kernel='rbf', gamma=settings['rbf_gamma'], cache_size=settings['cache'])
    else:
        #Assert kernel is poly
        classifier = svm.SVC(C=settings['poly_c'], kernel='poly', degree=settings['poly_d'], gamma=settings['poly_gamma'], coef0=settings['poly_r'], cache_size=settings['cache'])

    ''' train_features is a list of word bags
        train_samples is a list containing only 1s and -1s
            (corresponding to the class ie an MP's vote) '''
    
    complete_train_features = get_complete_bags(train_features, settings['entailment'])
    complete_test_features = get_complete_bags(test_features, settings['entailment'])

    classifier.fit(complete_train_features, train_samples)

    test_predictions = classifier.predict(complete_test_features)

    grouped_speeches = {}

    for member_id in settings['testing_mps']:
        grouped_speeches[member_id] = {
            'votes': members[member_id],
            'speeches': []
        }

    for i, feature in enumerate(test_features):
        grouped_speeches[feature['member']]['speeches'].append({
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

    print('Accuracy by MP: {}%'.format(100 * accuracy_score(member_votes, member_predictions)))
    print('F1 by MP: {}'.format(f1_score(member_votes, member_predictions)))

    print('Accuracy by speech: {}%'.format(100 * accuracy_score(test_samples, test_predictions)))
    print('F1 by speech: {}'.format(f1_score(test_samples, test_predictions)))

    return f1_score(test_samples, test_predictions)


def generate_linear_param_sets(linear_param_values):
    linear_param_sets = []
    for c_value in linear_param_values['cs']:
        linear_param_sets.append({
            'c': c_value
        })

    return linear_param_sets


def generate_rbf_param_sets(rbf_param_values):
    rbf_param_sets = []
    for c_value in rbf_param_values['cs']:
        for gamma_value in rbf_param_values['gammas']:
            rbf_param_sets.append({
                'c': c_value,
                'gamma': gamma_value
            })

    return rbf_param_sets


def generate_poly_param_sets(poly_param_values):
    poly_param_sets = []
    for c_value in poly_param_values['cs']:
        for gamma_value in poly_param_values['gammas']:
            for d_value in poly_param_values['ds']:
                for r_value in poly_param_values['rs']:
                    poly_param_sets.append({
                        'c': c_value,
                        'gamma': gamma_value,
                        'd': d_value,
                        'r': r_value
                    })

    return poly_param_sets


def generate_lower_log_params(max_param, no_of_params, log_diff):
    return logspace(log10(max_param) - (no_of_params - 1) * log10(log_diff), log10(max_param), no_of_params)


def generate_higher_log_params(min_param, no_of_params, log_diff):
    return logspace(log10(min_param), log10(min_param) + (no_of_params - 1) * log10(log_diff), no_of_params)


def generate_lower_params(max_param, no_of_params, diff):
    return [(diff * i) - (diff*(no_of_params - 1) - max_param) for i in range(no_of_params)]


def generate_higher_params(min_param, no_of_params, diff):
    return [(diff * i) + min_param for i in range(no_of_params)]


def find_linear_params(settings, mp_folds, linear_param_values):
    params_found = False
    while not params_found:
        print('Linear loop')
        params_found = True
        linear_param_sets = generate_linear_param_sets(linear_param_values)

        f1_matrix = array([compute_linear_fold_f1s(settings, fold, linear_param_sets) for fold in mp_folds]).transpose()

        max_f1_mean = -1

        for i, param_f1s in f1_matrix:
            if mean(param_f1s) > max_f1_mean:
                max_linear_param_set = linear_param_sets[i]
                max_f1s = param_f1s
                max_f1_mean = mean(param_f1s)

        if len(linear_param_values['cs']) > 1:
            min_c = linear_param_values['cs'][0]
            max_c = linear_param_values['cs'][len(linear_param_values['cs']) - 1]

            if max_linear_param_set['c'] == min_c:
                c_log_diff = linear_param_values['cs'][1] / linear_param_values['cs'][0]
                linear_param_values['cs'] = generate_lower_log_params(min_c, 5, c_log_diff)
                params_found = False
            elif max_linear_param_set['c'] == max_c:
                c_log_diff = linear_param_values['cs'][1] / linear_param_values['cs'][0]
                linear_param_values['cs'] = generate_higher_log_params(max_c, 5, c_log_diff)
                params_found = False

    return max_f1s, max_linear_param_set


def find_rbf_params(settings, mp_folds, rbf_param_values):
    params_found = False
    while not params_found:
        print('Rbf loop')
        params_found = True
        rbf_param_sets = generate_rbf_param_sets(rbf_param_values)

        f1_matrix = array([compute_rbf_fold_f1s(settings, fold, rbf_param_sets) for fold in mp_folds]).transpose()

        max_f1_mean = -1

        for i, param_f1s in f1_matrix:
            if mean(param_f1s) > max_f1_mean:
                max_rbf_param_set = rbf_param_sets[i]
                max_f1s = param_f1s
                max_f1_mean = mean(param_f1s)

        if len(rbf_param_values['cs']) > 1:
            min_c = rbf_param_values['cs'][0]
            max_c = rbf_param_values['cs'][len(rbf_param_values['cs']) - 1]

            if max_rbf_param_set['c'] == min_c:
                c_log_diff = rbf_param_values['cs'][1] / rbf_param_values['cs'][0]
                rbf_param_values['cs'] = generate_lower_log_params(min_c, 5, c_log_diff)
                params_found = False
            elif max_rbf_param_set['c'] == max_c:
                c_log_diff = rbf_param_values['cs'][1] / rbf_param_values['cs'][0]
                rbf_param_values['cs'] = generate_higher_log_params(max_c, 5, c_log_diff)
                params_found = False

        if len(rbf_param_values['gammas']) > 1:
            min_gamma = rbf_param_values['gammas'][0]
            max_gamma = rbf_param_values['gammas'][len(rbf_param_values['gammas']) - 1]

            if max_rbf_param_set['gamma'] == min_gamma:
                gamma_log_diff = rbf_param_values['gammas'][1] / rbf_param_values['gammas'][0]
                rbf_param_values['gammas'] = generate_lower_log_params(min_gamma, 5, gamma_log_diff)
                params_found = False
            elif max_rbf_param_set['gamma'] == max_gamma:
                gamma_log_diff = rbf_param_values['gammas'][1] / rbf_param_values['gammas'][0]
                rbf_param_values['gammas'] = generate_higher_log_params(max_gamma, 5, gamma_log_diff)
                params_found = False

    return max_f1s, max_rbf_param_set


def find_poly_params(settings, mp_folds, poly_param_values):
    params_found = False
    while not params_found:
        print('Poly loop')
        params_found = True
        poly_param_sets = generate_poly_param_sets(poly_param_values)

        f1_matrix = array([compute_poly_fold_f1s(settings, fold, poly_param_sets) for fold in mp_folds]).transpose()

        max_f1_mean = -1

        for i, param_f1s in f1_matrix:
            if mean(param_f1s) > max_f1_mean:
                max_poly_param_set = poly_param_sets[i]
                max_f1s = param_f1s
                max_f1_mean = mean(param_f1s)

        if len(poly_param_values['cs']) > 1:
            min_c = poly_param_values['cs'][0]
            max_c = poly_param_values['cs'][len(poly_param_values['cs']) - 1]

            if max_poly_param_set['c'] == min_c:
                c_log_diff = poly_param_values['cs'][1] / poly_param_values['cs'][0]
                poly_param_values['cs'] = generate_lower_log_params(min_c, 5, c_log_diff)
                params_found = False
            elif max_poly_param_set['c'] == max_c:
                c_log_diff = poly_param_values['cs'][1] / poly_param_values['cs'][0]
                poly_param_values['cs'] = generate_higher_log_params(max_c, 5, c_log_diff)
                params_found = False

        if len(poly_param_values['gammas']) > 1:
            min_gamma = poly_param_values['gammas'][0]
            max_gamma = poly_param_values['gammas'][len(poly_param_values['gammas']) - 1]

            if max_poly_param_set['gamma'] == min_gamma:
                gamma_log_diff = poly_param_values['gammas'][1] / poly_param_values['gammas'][0]
                poly_param_values['gammas'] = generate_lower_log_params(min_gamma, 5, gamma_log_diff)
                params_found = False
            elif max_poly_param_set['gamma'] == max_gamma:
                gamma_log_diff = poly_param_values['gammas'][1] / poly_param_values['gammas'][0]
                poly_param_values['gammas'] = generate_higher_log_params(max_gamma, 5, gamma_log_diff)
                params_found = False

        if len(poly_param_values['ds']) > 1:
            min_d = poly_param_values['ds'][0]
            max_d = poly_param_values['ds'][len(poly_param_values['ds']) - 1]

            if max_poly_param_set['d'] == min_d:
                d_diff = poly_param_values['ds'][1] - poly_param_values['ds'][0]
                poly_param_values['ds'] = generate_lower_params(min_d, 3, d_diff)
                params_found = False
            elif max_poly_param_set['d'] == max_d:
                d_diff = poly_param_values['ds'][1] - poly_param_values['ds'][0]
                poly_param_values['ds'] = generate_higher_params(max_d, 3, d_diff)
                params_found = False

        if len(poly_param_values['rs']) > 1:
            max_r = poly_param_values['rs'][len(poly_param_values['rs']) - 1]

            if max_poly_param_set['r'] == max_r:
                r_diff = poly_param_values['rs'][1] - poly_param_values['rs'][0]
                poly_param_values['rs'] = generate_higher_params(max_r, 3, r_diff)
                params_found = False

    return max_f1s, max_poly_param_set


def compute_t(differences):
    avg = sum(differences) / len(differences)

    squared_diff = 0

    for difference in differences:
        squared_diff += pow(difference - avg, 2)

    variance = squared_diff / (len(differences) - 1)

    return avg / (math.sqrt(variance / len(differences)))


def change_n_gram(settings, increment, current_f1s, mp_folds):


    significant_change = settings['n_gram'] + increment in range(1, 10)

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


def generate_linear_values(no_of_cs):
    c_values = logspace(-3, 3, no_of_cs)

    return {
        'cs': c_values
    }


def generate_rbf_values(no_of_cs, no_of_gammas):
    c_values = logspace(-3, 3, no_of_cs)
    gamma_values = logspace(-3, 3, no_of_gammas)

    return {
        'cs': c_values,
        'gammas': gamma_values
    }


def generate_poly_values(no_of_cs, no_of_gammas, no_of_ds, no_of_rs):
    c_values = logspace(-3, 3, no_of_cs)
    gamma_values = logspace(-3, 3, no_of_gammas)
    d_values = [i for i in range(2, 2 + no_of_ds)]
    r_values = [i for i in range(0, no_of_rs)]

    return {
        'cs': c_values,
        'gammas': gamma_values,
        'ds': d_values,
        'rs': r_values
    }


def generate_refined_linear_values(linear_params, no_of_cs):

    c_values = logspace(log10(linear_params['c']) - 1, log10(linear_params['c']) + 1, no_of_cs)

    return {
        'cs': c_values
    }


def generate_refined_rbf_values(rbf_params, no_of_cs, no_of_gammas):

    c_values = logspace(log10(rbf_params['c']) - 1, log10(rbf_params['c']) + 1, no_of_cs)

    gamma_values = logspace(log10(rbf_params['gamma']) - 1, log10(rbf_params['gamma']) + 1, no_of_gammas)

    return {
        'cs': c_values,
        'gammas': gamma_values
    }


def generate_refined_poly_values(poly_params, no_of_cs, no_of_gammas, no_of_ds, no_of_rs):

    c_values = logspace(log10(poly_params['c']) - 1, log10(poly_params['c']) + 1, no_of_cs)

    gamma_values = logspace(log10(poly_params['gamma']) - 1, log10(poly_params['gamma']) + 1, no_of_gammas)

    d_values = [i for i in range(poly_params['d'], poly_params['d'] + no_of_ds)]

    r_values = [i for i in range(poly_params['r'], poly_params['r'] + no_of_rs)]

    return {
        'cs': c_values,
        'gammas': gamma_values,
        'ds': d_values,
        'rs': r_values
    }


def refine_linear_params(settings, mp_folds):

    linear_param_values = generate_linear_values(7)

    _, linear_params = find_linear_params(settings, mp_folds, linear_param_values)

    linear_param_values = generate_refined_linear_values(linear_params, 7)

    return find_linear_params(settings, mp_folds, linear_param_values)


def refine_rbf_params(settings, mp_folds):

    rbf_param_values = generate_rbf_values(7, 7)

    _, rbf_params = find_rbf_params(settings, mp_folds, rbf_param_values)

    rbf_param_values = generate_refined_rbf_values(rbf_params, 7, 7)

    return find_rbf_params(settings, mp_folds, rbf_param_values)


def refine_poly_params(settings, mp_folds):

    poly_param_values = generate_poly_values(7, 7, 3, 1)

    _, poly_params = find_poly_params(settings, mp_folds, poly_param_values)

    poly_param_values = generate_refined_poly_values(poly_params, 7, 7, 1, 1)

    return find_poly_params(settings, mp_folds, poly_param_values)


def learn_settings(settings, mp_folds):

    linear_param_values = generate_linear_values(7)

    linear_f1s, linear_params = find_linear_params(settings, mp_folds, linear_param_values)

    linear_mean = mean(linear_f1s)

    print('Linear mean: {}'.format(linear_mean))

    rbf_param_values = generate_rbf_values(7, 7)

    rbf_f1s, rbf_params = find_rbf_params(settings, mp_folds, rbf_param_values)

    rbf_mean = mean(rbf_f1s)

    print('RBF mean: {}'.format(rbf_mean))

    poly_param_values = generate_poly_values(7, 7, 1, 1)

    poly_f1s, poly_params = find_poly_params(settings, mp_folds, poly_param_values)

    poly_mean = mean(poly_f1s)

    print('Poly mean: {}'.format(poly_mean))

    if linear_mean > rbf_mean:
        if linear_mean > poly_mean:
            settings['kernel'] = 'linear'
            settings['linear_c'] = linear_params['c']
            current_f1s = linear_f1s
        else:
            settings['kernel'] = 'poly'
            settings['poly_c'] = poly_params['c']
            settings['poly_gamma'] = poly_params['gamma']
            settings['poly_d'] = poly_params['d']
            settings['poly_r'] = poly_params['r']
            current_f1s = poly_f1s
    else:
        if rbf_mean > poly_mean:
            settings['kernel'] = 'rbf'
            settings['rbf_c'] = rbf_params['c']
            settings['rbf_gamma'] = rbf_params['gamma']
            current_f1s = rbf_f1s
        else:
            settings['kernel'] = 'poly'
            settings['poly_c'] = poly_params['c']
            settings['poly_gamma'] = poly_params['gamma']
            settings['poly_d'] = poly_params['d']
            settings['poly_r'] = poly_params['r']
            current_f1s = poly_f1s

    print('Hyper parameters learned')
    print(current_f1s)

    current_f1s = choose_boolean_setting(settings, 'normalise', current_f1s, mp_folds)

    current_f1s = choose_boolean_setting(settings, 'remove_stopwords', current_f1s, mp_folds)

    current_f1s = choose_boolean_setting(settings, 'stem_words', current_f1s, mp_folds)

    current_f1s = choose_boolean_setting(settings, 'group_numbers', current_f1s, mp_folds)

    print('Boolean settings learned')
    print(current_f1s)

    current_f1s = change_n_gram(settings, 1, current_f1s, mp_folds)

    print('N gram learned')
    print(current_f1s)

    if settings['kernel'] == 'linear':
        current_f1s, linear_params = refine_linear_params(settings, mp_folds)
        settings['linear_c'] = linear_params['c']

    elif settings['kernel'] == 'rbf':
        current_f1s, rbf_params = refine_rbf_params(settings, mp_folds)
        settings['rbf_c'] = rbf_params['c']
        settings['rbf_gamma'] = rbf_params['gamma']

    else:
        #Assert kernel = poly
        current_f1s, poly_params = refine_poly_params(settings, mp_folds)
        settings['poly_c'] = poly_params['c']
        settings['poly_gamma'] = poly_params['gamma']
        settings['poly_d'] = poly_params['d']
        settings['poly_r'] = poly_params['r']

    settings['max_bag_size'] = False

    pickle.dump(settings, open('settings.p', 'wb'))

    print('Hyper parameters refined')
    print(current_f1s)
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
        'test_division': 102565,
        'all_debates': False,
        'debate_terms': ['iraq', 'terrorism', 'middle east', 'defence policy',
                         'defence in the world', 'afghanistan'],
        'no_of_folds': 10,
        'entailment': True,
        'normalise': False,
        'division_ids': [102564, 102565],
        'test_mp_file': 'test_data_combined.txt',
        'train_mp_file': 'train_data_combined.txt',
        'cache': 1024
    }

    settings['debates'] = get_debates(settings)

    settings['testing_mps'] = get_members_from_file(settings['test_mp_file'])

    settings['training_mps'] = get_members_from_file(settings['train_mp_file'])

    mp_folds = get_mp_folds(settings)

    print('Made splits')

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

    print('Normalisation: {}'.format(settings['normalise']))
    print('N-gram: {}'.format(settings['n_gram']))
    print('Remove stopwords: {}'.format(settings['remove_stopwords']))
    print('Stem words: {}'.format(settings['stem_words']))
    print('Group numbers: {}'.format(settings['group_numbers']))
    print()


    data = {
        'train': train_data,
        'test': test_data
    }

    compute_member_f1s(settings, data)

run()
