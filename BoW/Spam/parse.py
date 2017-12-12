""" Parses the spam email dataset """

import collections
import re

from bs4 import BeautifulSoup


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


def parse_ems(file):
    """ Parses an ems file and creates the corresponding bags of words"""
    handler = open('Data/Spam/{}.ems'.format(file)).read()
    soup = BeautifulSoup(handler, "lxml-xml")
    messages = soup.find_all("MESSAGE")
    bags = []
    sum_bag = collections.Counter()
    for message in messages:
        word_list = generate_word_list(str(message.MESSAGE_BODY))
        count = collections.Counter()
        for word in word_list:
            count[word] += 1
            sum_bag[word] += 1
        bags.append(count)
    return sum_bag


gen_bag = parse_ems('Data/Spam/train_GEN.ems')
spam_bag = parse_ems('Data/Spam/train_SPAM.ems')