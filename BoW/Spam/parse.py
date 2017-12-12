""" Parses the spam email dataset """

import collections
import re

from bs4 import BeautifulSoup
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


def parse_ems(file, sum_bag):
    """ Parses an ems file and creates the corresponding bags of words"""
    handler = open(file).read()
    soup = BeautifulSoup(handler, "lxml-xml")
    messages = soup.find_all("MESSAGE")
    bags = []
    for message in messages:
        word_list = generate_word_list(str(message.MESSAGE_BODY))
        bag = collections.Counter()
        for word in word_list:
            bag[word] += 1
            sum_bag[word] += 1
        bags.append(bag)
    return bags, sum_bag


def condense_bags(bags, words):
    """ Returns an array of integer arrays containing the counts of the words
         (in the array provided) and an array of the Counter bags """
    return [[bag[word] for word in words] for bag in bags]


def generate_classifier_data(gen_file, spam_file, num_of_words):
    """ Returns the features and samples in a form that can be used
         by a classifier, given the filenames for the data """

    gen_bags, sum_bag = parse_ems(gen_file, collections.Counter())
    spam_bags, sum_bag = parse_ems(spam_file, sum_bag)


    common_words = [word[0] for word in sum_bag.most_common(num_of_words)]

    condensed_gen_bags = condense_bags(gen_bags, common_words)
    condensed_spam_bags = condense_bags(spam_bags, common_words)

    features = condensed_gen_bags + condensed_spam_bags
    samples = []
    for _ in range(len(condensed_gen_bags)):
        samples.append(1)

    for _ in range(len(condensed_spam_bags)):
        samples.append(-1)

    return features, samples


def run():
    """ If this has a green line I'll get annoyed """
    for count in range(1, 100):
        train_features, train_samples = generate_classifier_data('Data/Spam/train_GEN.ems',
                                                                 'Data/Spam/train_SPAM.ems', count)

        test_features, test_samples = generate_classifier_data('Data/Spam/test_GEN.ems',
                                                               'Data/Spam/test_SPAM.ems', count)

        classifier = svm.SVC()
        classifier.fit(train_features, train_samples)
        print('{} words score: {}'.format(count, classifier.score(test_features, test_samples)))

run()
