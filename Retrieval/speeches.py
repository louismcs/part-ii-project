"""Retrieves the texts of all relevant House of Commons speeches"""

import json
import sqlite3
from urllib.request import urlopen
import re
import Levenshtein
import requests
from bs4 import BeautifulSoup
from helper import (date_range, DB_PATH, START_DATE, END_DATE,
                    generate_debates_csv, generate_speeches_csv)


class MatchException(Exception):
    """ Raised when there is no appropriate match in the database for a given speaker """
    pass


def generate_name_list(curs):
    """ Generates a python dictionary mapping mp names to ids """
    curs.execute("SELECT ID, FULL_NAME FROM MEMBER")
    rows = curs.fetchall()
    return rows


def insert_debate(conn, curs, url, day, title):
    """ Inserts a debate into the database given its data """
    try:
        curs.execute("INSERT INTO DEBATE (URL, DATE, TITLE) VALUES (?, ?, ?)",
                     (url, day.strftime('%Y-%m-%d'), title))
        conn.commit()
    except sqlite3.OperationalError:
        print('FAILED DEBATE INSERT: {} - {} - {}'.format(url, day.strftime('%Y-%m-%d'), title))


def insert_speech(conn, curs, url, member_id, quote):
    """ Inserts a speech into the database given its data """
    try:
        curs.execute("INSERT INTO SPEECH (DEBATE_URL, MEMBER_ID, QUOTE) VALUES (?, ?, ?)",
                     (url, member_id, quote))
        conn.commit()
    except sqlite3.OperationalError:
        print('FAILED SPEECH INSERT: {} - {} - {}'.format(url, member_id, quote))

def remove_title(name):
    """ Removes the first title prefix from the given string """
    titles = ['Mr ', 'Ms ', 'Mrs ', 'Miss ', 'Dr ', 'Professor ', 'Reverend ', 'Sir ', 'Dame ', 'Hon. ', 'Hon ']

    for title in titles:
        if name.startswith(title):
            name = name[len(title):]

    return name


def remove_titles(name):
    """ Removes title prefixes from the given string """
    ret = remove_title(name)
    while ret != name:
        name = ret
        ret = remove_title(name)

    return ret


def match_name(speaker, name_list, match_list, black_list):
    """ Returns the member_id for the given speaker where a match exists """
    if speaker in match_list:
        mp_id = match_list[speaker]
    elif speaker in black_list:
        raise MatchException()
    else:
        max_similarity = 0
        no_titles = remove_titles(speaker)
        for name in name_list:
            similarity = Levenshtein.ratio(remove_titles(name[1]), no_titles)
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = name

        if max_similarity > 0.9:
            name_list.remove(best_match)
            mp_id = best_match[0]
            match_list[speaker] = mp_id
        else:
            black_list.append(speaker)
            with open("blacklist.txt", "a") as myfile:
                myfile.write("{};{};{}\n".format(no_titles, remove_titles(best_match[1]), max_similarity))
            raise MatchException()

    return mp_id

def get_paragraph_text(paragraph):
    """Converts a paragraph tag to plain text"""
    paragraph = re.sub(r"<p.*?>", "", paragraph)
    paragraph = re.sub(r"</p.*?>", "", paragraph)
    paragraph = re.sub(r"<a.*?>.*?</a>", "", paragraph)
    paragraph = re.sub(r"<span.*?>.*?</span>", "", paragraph)
    paragraph = re.sub(r"<q.*?>.*?</q>", "\"", paragraph)
    return paragraph


def add_quote(blockquote, url, name_list, match_list, black_list, conn, curs):
    """Adds a quote (identified by its html element) to the database"""
    try:
        speaker = blockquote.cite.a['title']
        paragraphs = blockquote.find_all("p")
        quote = ""
        for paragraph in paragraphs:
            quote += get_paragraph_text(str(paragraph)) + "\n"
        try:
            member_id = match_name(speaker, name_list, match_list, black_list)
            insert_speech(conn, curs, url, member_id, quote)
        except MatchException:
            pass
    except TypeError:
        print('Cannot parse quote')


def add_debate(url, day, title, name_list, match_list, black_list, conn, curs):
    """Adds the speeches from a debate (identified by its url) to the database"""
    insert_debate(conn, curs, url, day, title)
    print('Debate: {}'.format(title))
    page = urlopen(url)
    page_soup = BeautifulSoup(page, "html.parser")
    blockquotes = page_soup.find_all("blockquote")
    for blockquote in blockquotes:
        add_quote(blockquote, url, name_list, match_list, black_list, conn, curs)


def add_day(day, name_list, match_list, black_list, conn, curs):
    """Gets the speeches for a given day"""
    date_string = day.strftime("%Y/%b/%d").lower()
    url = 'http://hansard.millbanksystems.com/sittings/{}.js'.format(date_string)
    res = requests.get(url)
    try:
        obj = json.loads(res.text)
        try:
            sections = obj[0]['house_of_commons_sitting']['top_level_sections']
            for section in sections:
                try:
                    sec = section['section']
                    add_debate('http://hansard.millbanksystems.com/commons/{}/{}'
                                .format(date_string, sec['slug']), day, sec['title'],
                                name_list, match_list, black_list, conn, curs)
                except KeyError:
                    print('Not a standard section')
        except KeyError:
            print('Not standard sections')

    except ValueError:
        print('No data for {}'.format(date_string))


def get_speeches():
    """ Adds debates and their speeches to the database """
    conn = sqlite3.connect(DB_PATH)
    curs = conn.cursor()
    name_list = generate_name_list(curs)
    match_list = {}
    black_list = []
    for day in date_range(START_DATE, END_DATE):
        add_day(day, name_list, match_list, black_list, conn, curs)
    generate_debates_csv()
    generate_speeches_csv()
