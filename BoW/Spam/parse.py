""" Learns to detect spam using the spam email dataset """

import collections
import re

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk import PorterStemmer
from nltk import ngrams
from sklearn import svm


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


def get_n_grams(word_list, gram_size):
    """ Given a word list and some gram size, returns a list of all n grams for n <= gram_size """
    if gram_size == 1:
        ret = word_list
    else:
        ret = ngrams(word_list, gram_size) + get_n_grams(word_list, gram_size - 1)

    return ret


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

    return get_n_grams(word_list, settings['n_gram'])


def get_messages(file):
    """ Returns the BeautifulSoup of all messages in the given .ems file """
    handler = open(file).read()
    soup = BeautifulSoup(handler, "lxml-xml")
    return soup.find_all("MESSAGE")


def parse_train_ems(file, settings, sum_bag):
    """ Parses a train ems file and creates the corresponding bags of words"""
    messages = get_messages(file)
    bags = []
    for message in messages:
        word_list = generate_word_list(str(message.MESSAGE_BODY), settings)
        bag = collections.Counter()
        for word in word_list:
            bag[word] += 1
            sum_bag[word] += 1
        bags.append(bag)
    return bags, sum_bag


def parse_test_ems(file, settings):
    """ Parses a test ems file and creates the corresponding bags of words"""
    messages = get_messages(file)
    bags = []
    for message in messages:
        word_list = generate_word_list(str(message.MESSAGE_BODY), settings)
        bag = collections.Counter()
        for word in word_list:
            bag[word] += 1
        bags.append(bag)
    return bags


def condense_bags(bags, words):
    """ Returns an array of integer arrays containing the counts of the words
         (in the array provided) and an array of the Counter bags """
    return [[bag[word] for word in words] for bag in bags]


def generate_classifier_data(gen_bags, spam_bags, common_words):
    """ Returns the features and samples in a form that can be used
         by a classifier, given the bags and most common words in them """
    condensed_gen_bags = condense_bags(gen_bags, common_words)
    condensed_spam_bags = condense_bags(spam_bags, common_words)
    features = condensed_gen_bags + condensed_spam_bags
    samples = []
    for _ in range(len(condensed_gen_bags)):
        samples.append(1)

    for _ in range(len(condensed_spam_bags)):
        samples.append(-1)

    return features, samples


def generate_train_data(gen_file, spam_file, settings):
    """ Returns the features and samples in a form that can be used
         by a classifier, given the filenames for the data """

    gen_bags, sum_bag = parse_train_ems(gen_file, settings, collections.Counter())
    spam_bags, sum_bag = parse_train_ems(spam_file, settings, sum_bag)

    common_words = [word[0] for word in sum_bag.most_common(settings['bag_size'])]

    features, samples = generate_classifier_data(gen_bags, spam_bags, common_words)

    return features, samples, common_words


def generate_test_data(gen_file, spam_file, common_words, settings):
    """ Returns the features and samples in a form that can be used
         by a classifier, given the filenames for the data """

    gen_bags = parse_test_ems(gen_file, settings)
    spam_bags = parse_test_ems(spam_file, settings)

    features, samples = generate_classifier_data(gen_bags, spam_bags, common_words)

    return features, samples


def run():
    """ Sets the settings and runs the program  """

    settings = {
        'black_list': [],
        'white_list': [],
        'bag_size': 100,
        'remove_stopwords': False,
        'stem_words': True,
        'n_gram': 2
    }

    train_features, train_samples, common_words = generate_train_data('Data/Spam/train_GEN.ems',
                                                                      'Data/Spam/train_SPAM.ems',
                                                                      settings)

    test_features, test_samples = generate_test_data('Data/Spam/test_GEN.ems',
                                                     'Data/Spam/test_SPAM.ems',
                                                     common_words, settings)

    classifier = svm.SVC()
    classifier.fit(train_features, train_samples)
    print('Score: {}'.format(classifier.score(test_features, test_samples)))
    print(common_words)

run()
