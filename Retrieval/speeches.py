"""Retrieves the texts of all relevant House of Commons speeches"""

import json
from datetime import timedelta, date
from urllib.request import urlopen
import re
import requests
from bs4 import BeautifulSoup

def date_range(start_date, end_date):
    """Retuns all dates between start_date (inclusive) and end_date (exclusive)"""
    for count in range(int((end_date - start_date).days)):
        yield start_date + timedelta(count)


def get_paragraph_text(paragraph):
    """Converts a paragraph tag to plain text"""
    paragraph = re.sub(r"<p.*?>", "", paragraph)
    paragraph = re.sub(r"</p.*?>", "", paragraph)
    paragraph = re.sub(r"<a.*?>.*?</a>", "", paragraph)
    paragraph = re.sub(r"<span.*?>.*?</span>", "", paragraph)
    paragraph = re.sub(r"<q.*?>.*?</q>", "\"", paragraph)
    return paragraph

def add_quote(blockquote, url, day):
    """Adds a quote (identified by its html element) to the database"""
    try:
        speaker = blockquote.cite.a['title']
        paragraphs = blockquote.find_all("p")
        quote = ""
        for paragraph in paragraphs:
            quote += get_paragraph_text(str(paragraph)) + "\n"
    except TypeError:
        print('Cannot parse quote')


def add_debate(url, day):
    """Adds the speeches from a debate (identified by its url) to the database"""
    page = urlopen(url)
    page_soup = BeautifulSoup(page, "html.parser")
    blockquotes = page_soup.find_all("blockquote")
    for blockquote in blockquotes:
        add_quote(blockquote, url, day)

def add_day(day):
    """Gets the speeches for a given day"""
    date_string = day.strftime("%Y/%b/%d")
    url = 'http://hansard.millbanksystems.com/sittings/{}.js'.format(date_string)
    res = requests.get(url)
    try:
        obj = json.loads(res.text)
        sections = obj[0]['house_of_commons_sitting']['top_level_sections']
        for section in sections:
            try:
                sec = section['section']
                if 'iraq' in sec['title'].lower():
                    add_debate('http://hansard.millbanksystems.com/commons/{}/{}'
                               .format(date_string, sec['slug']), day)
            except KeyError:
                print('Not a standard section')


    except ValueError:
        print('No data for {}'.format(date_string))

def run():
    ''' add_debate('http://hansard.millbanksystems.com/commons/2003/mar/18/iraq') '''
    start_date = date(2001, 9, 11)
    end_date = date(2003, 3, 19)
    for day in date_range(start_date, end_date):
        add_day(day)

run()