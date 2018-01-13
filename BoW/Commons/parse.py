""" Learns to guess how an MP votes, given transcriptions of House of Commons Sittings """

import collections
import re

from nltk.corpus import stopwords
from nltk import PorterStemmer
from nltk import ngrams
from sklearn import svm

def get_train_speeches(db_path, settings):
    """ Returns all the speeches in the given database that match the given settings """

    """ speeches only contains speeches of MPs that voted in the division given in settings
        should only use MPs used for training
        speech.text is text
        speech.aye is a boolean giving how the mp voted"""
    speeches = []
    return speeches


def remove_tags(body):
    """ Removes any xml tags from a given xml body """
    return re.sub(r"<(.)*?>", "", body)


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


def generate_word_list(body, settings):
    """ Returns a list of words, given a message tag """
    body = remove_tags(body)
    body = remove_punctuation(body)
    body = body.lower()

    word_list = body.split()

    if settings['remove_stopwords']:
        word_list = remove_stopwords(word_list, settings['black_list'], settings['white_list'])

    if settings['stem_words']:
        word_list = stem_words(word_list)

    return ngrams(word_list, settings['n_gram'])

def parse_train_ems(db_path, settings):
    """ Parses a train ems file and creates the corresponding bags of words"""

    speeches = get_train_speeches(db_path, settings)

    aye_bags = []
    no_bags = []

    sum_bag = collections.Counter()

    for speech in speeches:
        word_list = generate_word_list(speech.text, settings)
        bag = collections.Counter()
        for word in word_list:
            bag[word] += 1
            sum_bag[word] += 1

        if speech.aye:
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

def generate_train_data(db_path, settings):
    """ Returns the features and samples in a form that can be used
         by a classifier, given the filenames for the data """

    aye_bags, no_bags, sum_bag = parse_train_ems(db_path, settings)

    common_words = [word[0] for word in sum_bag.most_common(settings['bag_size'])]

    features, samples = generate_classifier_data(aye_bags, no_bags, common_words)

    return features, samples, common_words


def get_test_speeches(db_path, settings):
    """ Returns all the speeches in the given database that match the given settings """

    speeches = []
    return speeches


def parse_test_ems(db_path, settings):
    """ Parses a test ems file and creates the corresponding bags of words"""

    speeches = get_test_speeches(db_path, settings)

    aye_bags = []
    no_bags = []

    for speech in speeches:
        word_list = generate_word_list(speech.text, settings)
        bag = collections.Counter()
        for word in word_list:
            bag[word] += 1

        if speech.aye:
            aye_bags.append(bag)
        else:
            no_bags.append(bag)

    return aye_bags, no_bags

def generate_test_data(db_path, common_words, settings):
    """ Returns the features and samples in a form that can be used
         by a classifier, given the filenames for the data """

    aye_bags, no_bags = parse_test_ems(db_path, settings)

    features, samples = generate_classifier_data(aye_bags, no_bags, common_words)

    return features, samples


def run():
    """ Sets the settings and runs the program  """

    settings = {
        'black_list': [],
        'white_list': [],
        'bag_size': 100,
        'remove_stopwords': False,
        'stem_words': False,
        'n_gram': 1,
        'division_id': 102564,
        'all_debates': True,
        'debate_terms': []
    }
    ''' Make a for loop where settings.test_mps changes in each loop for cross validation '''
    ''' Try to make it work with words included in the debates rather than just titles? '''

    train_features, train_samples, common_words = generate_train_data('Data/Corpus/database.db',
                                                                      settings)

    test_features, test_samples = generate_test_data('Data/Corpus/database.db',
                                                     common_words, settings)

    classifier = svm.SVC()
    ''' train_features is a list of word bags
        train_samples is a list containing only 1s and -1s
            (corresponding to the class ie an MP's vote) '''
    classifier.fit(train_features, train_samples)
    print('Score: {}'.format(classifier.score(test_features, test_samples)))
    print(common_words)

run()
