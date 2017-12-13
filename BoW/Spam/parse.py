""" Parses the spam email dataset """

import collections
import re

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
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

def generate_word_list(body):
    """ Returns a list of words, given a message tag """
    body = remove_tags(body)
    body = remove_punctuation(body)
    body = body.lower()
    return body.split()


def remove_stopwords(word_list, black_list, white_list):
    """ Returns a list of words (with stop words removed), given a word list """
    stop = set(stopwords.words('english'))
    return [word for word in word_list
            if (word in white_list) or ((word not in stop) and (word not in black_list))]


def parse_train_ems(file, black_list, white_list, sum_bag):
    """ Parses a train ems file and creates the corresponding bags of words"""
    handler = open(file).read()
    soup = BeautifulSoup(handler, "lxml-xml")
    messages = soup.find_all("MESSAGE")
    bags = []
    for message in messages:
        word_list = generate_word_list(str(message.MESSAGE_BODY))
        #word_list = remove_stopwords(word_list, black_list, white_list)
        bag = collections.Counter()
        for word in word_list:
            bag[word] += 1
            sum_bag[word] += 1
        bags.append(bag)
    return bags, sum_bag


def parse_test_ems(file, black_list, white_list):
    """ Parses a test ems file and creates the corresponding bags of words"""
    handler = open(file).read()
    soup = BeautifulSoup(handler, "lxml-xml")
    messages = soup.find_all("MESSAGE")
    bags = []
    for message in messages:
        word_list = generate_word_list(str(message.MESSAGE_BODY))
        word_list = remove_stopwords(word_list, black_list, white_list)
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

    condensed_gen_bags = condense_bags(gen_bags, common_words)
    condensed_spam_bags = condense_bags(spam_bags, common_words)
    features = condensed_gen_bags + condensed_spam_bags
    samples = []
    for _ in range(len(condensed_gen_bags)):
        samples.append(1)

    for _ in range(len(condensed_spam_bags)):
        samples.append(-1)

    return features, samples


def generate_train_data(gen_file, spam_file, num_of_words, black_list, white_list):
    """ Returns the features and samples in a form that can be used
         by a classifier, given the filenames for the data """

    gen_bags, sum_bag = parse_train_ems(gen_file, black_list, white_list, collections.Counter())
    spam_bags, sum_bag = parse_train_ems(spam_file, black_list, white_list, sum_bag)

    common_words = [word[0] for word in sum_bag.most_common(num_of_words)]

    features, samples = generate_classifier_data(gen_bags, spam_bags, common_words)

    return features, samples, common_words


def generate_test_data(gen_file, spam_file, common_words, black_list, white_list):
    """ Returns the features and samples in a form that can be used
         by a classifier, given the filenames for the data """

    gen_bags = parse_test_ems(gen_file, black_list, white_list)
    spam_bags = parse_test_ems(spam_file, black_list, white_list)

    features, samples = generate_classifier_data(gen_bags, spam_bags, common_words)

    return features, samples


def run():
    """ If this has a green line I'll get annoyed """

    black_list = ['#name', '#num', '#website', '#char']
    white_list = []

    for count in range(1, 100):
        train_features, train_samples, common_words = generate_train_data('Data/Spam/train_GEN.ems',
                                                                          'Data/Spam/train_SPAM.ems',
                                                                          count, black_list, white_list)

        test_features, test_samples = generate_test_data('Data/Spam/test_GEN.ems',
                                                         'Data/Spam/test_SPAM.ems', common_words, black_list, white_list)

        classifier = svm.SVC()
        classifier.fit(train_features, train_samples)
        print('{} words score: {}'.format(count, classifier.score(test_features, test_samples)))
        print(common_words)
        print('----------')

run()
