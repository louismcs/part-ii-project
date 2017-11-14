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
    curs.execute("SELECT ID, FULL_NAME, GIVEN_NAME, FAMILY_NAME FROM MEMBER")
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
    titles = ['Mr ', 'Ms ', 'Mrs ', 'Miss ', 'Dr ', 'Professor ',
              'Reverend ', 'Sir ', 'Dame ', 'Hon. ', 'Hon ']

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


def match_first_and_family_name(no_titles, name_list, black_list, speaker):
    """ Returns the member id for mps using their given and family names """
    max_similarity = 0
    for name in name_list:
        similarity = Levenshtein.ratio('{} {}'.format(name[2], name[3]), no_titles)
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = name

    if max_similarity > 0.85:
        name_list.remove(best_match)
        return best_match[0]
    else:
        black_list.append(speaker)
        with open("blacklist.txt", "a") as myfile:
            myfile.write("{};{};{}\n".format(no_titles,
                                             '{} {}'.format(best_match[2], best_match[3]),
                                             max_similarity))
        raise MatchException()

def match_full_name(speaker, name_list, match_list, black_list):
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

        if max_similarity > 0.85:
            name_list.remove(best_match)
            mp_id = best_match[0]
        else:
            mp_id = match_first_and_family_name(no_titles, name_list, black_list, speaker)

        match_list[speaker] = mp_id

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
            member_id = match_full_name(speaker, name_list, match_list, black_list)
            insert_speech(conn, curs, url, member_id, quote)
            print('{} - MATCH FOUND!'.format(speaker))
        except MatchException:
            pass
    except TypeError:
        print('Cannot parse quote')


def add_debate(url, day, title, name_list, match_list, black_list, conn, curs, to_do_list):
    """Adds the speeches from a debate (identified by its url) to the database"""
    #insert_debate(conn, curs, url, day, title)
    print('Debate: {} - {}'.format(title, day.strftime("%Y/%b/%d")))
    page = urlopen(url)
    page_soup = BeautifulSoup(page, "html.parser")
    blockquotes = page_soup.find_all("blockquote")
    for blockquote in blockquotes:
        try:
            if remove_titles(blockquote.cite.a['title']) in to_do_list:
                add_quote(blockquote, url, name_list, match_list, black_list, conn, curs)
        except TypeError:
            pass

def add_day(day, name_list, match_list, black_list, conn, curs, to_do_list):
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
                               name_list, match_list, black_list, conn, curs, to_do_list)
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
    to_do_list = ["David Blunkett",
                  "Michael Martin",
                  "Patrick Cormack",
                  "Michael Howard",
                  "Matthew Taylor",
                  "John McFall",
                  "Don Foster",
                  "Brian Mawhinney",
                  "Brian Donohoe",
                  "Anne McIntosh",
                  "Andrew Lansley",
                  "Paul Tyler",
                  "John Hutton",
                  "Bob Marshall-Andrews",
                  "Menzies Campbell",
                  "Oona King",
                  "Earl of Ancram",
                  "Jenny Tonge",
                  "Francis Maude",
                  "Peter Hain",
                  "Nigel Evans",
                  "Mick Clapham",
                  "Geoff Hoon",
                  "Margaret Jackson",
                  "Malcolm Bruce",
                  "Archy Kirkwood",
                  "George Young",
                  "Martin Smyth",
                  "Angela Browning",
                  "Clive Soley",
                  "Andrew Stunell",
                  "Jack Cunningham",
                  "Jim Paice",
                  "Tony Banks",
                  "Douglas Hogg",
                  "Andrew Robathan",
                  "John Reid",
                  "Quentin Davies",
                  "Ian Paisley",
                  "Andrew King",
                  "Derek Foster",
                  "John Burnett",
                  "Helen Brinton",
                  "George Foulkes",
                  "Ian Lucas",
                  "Howard Flight",
                  "Alan J. Williams",
                  "Gregory Barker",
                  "Donald Anderson",
                  "Michael Lord",
                  "Richard Allan",
                  "David Borrow",
                  "John Horam",
                  "Tim Boswell",
                  "Phil Willis",
                  "Peter Mandelson",
                  "Lady Sylvia Hermon",
                  "David Willetts",
                  "Alistair Darling",
                  "David MacLean",
                  "James Arbuthnot",
                  "Joyce Quin",
                  "John Gummer",
                  "Lindsay Hoyle",
                  "Beverley Hughes",
                  "Alan Beith",
                  "John Maples",
                  "Alan Howarth",
                  "Michael Spicer",
                  "Martin O'Neill",
                  "Tony Wright",
                  "Richard Spring",
                  "John Austin-Walker",
                  "James Wray",
                  "Jim Knight",
                  "Bill Cash",
                  "Tessa Jowell",
                  "Dennis Turner",
                  "Keith Bradley",
                  "Virginia Bottomley",
                  "Chris Smith",
                  "Douglas Naysmith",
                  "Jean Corston",
                  "Paul Boateng",
                  "Gillian Shephard",
                  "David Chidgey",
                  "Estelle Morris",
                  "Eddie O'Hara",
                  "Paul Murphy",
                  "Don Touhig",
                  "Jimmy Hood",
                  "Helen Liddell",
                  "Brian Cotter",
                  "Michael Wills",
                  "Andy Love",
                  "William Hague",
                  "David Trimble",
                  "Lewis Moonie",
                  "Ann Taylor",
                  "Des Browne",
                  "Des Turner",
                  "John Prescott",
                  "Dave Watts",
                  "Irene Adams",
                  "Nick Brown",
                  "Anne Picking",
                  "Dawn Primarolo"]
    for day in date_range(START_DATE, END_DATE):
        add_day(day, name_list, match_list, black_list, conn, curs, to_do_list)
    generate_debates_csv()
    generate_speeches_csv()

get_speeches()
