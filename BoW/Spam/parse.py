""" Parses the spam email dataset """

import collections
import re
import string

from bs4 import BeautifulSoup


def parse_ems():
    """ Parses the ems email files """
    handler = open('Data/Spam/train_GEN.ems').read()
    soup = BeautifulSoup(handler, "lxml-xml")
    messages = soup.find_all("MESSAGE")
    bags = []
    sum_bag = collections.Counter()
    for message in messages:
        body = str(message.MESSAGE_BODY)
        body = re.sub(r"<(.)*?>", "", body)
        body = body.replace("\n", " ")
        body = re.sub(r"[^\w\d\s#']", '', body)
        body = body.replace(" '", " ")
        body = body.replace("' ", " ")
        body = body.replace(" -", " ")
        body = body.replace("- ", " ")
        body = body.lower()
        word_list = body.split()
        count = collections.Counter()
        for word in word_list:
            count[word] += 1
            sum_bag[word] += 1
        bags.append(count)

    print(sum_bag.most_common(100))
parse_ems()
