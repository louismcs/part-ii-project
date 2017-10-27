"""Retrieves the texts of all relevant House of Commons speeches"""

import json
from urllib.request import urlopen
import re
import requests
from bs4 import BeautifulSoup

def get_paragraph_text(paragraph):
    """Converts a paragraph tag to plain text"""
    paragraph = re.sub(r"<p.*?>", "", paragraph)
    paragraph = re.sub(r"</p.*?>", "", paragraph)
    paragraph = re.sub(r"<a.*?>.*?</a>", "", paragraph)
    paragraph = re.sub(r"<span.*?>.*?</span>", "", paragraph)
    paragraph = re.sub(r"<q.*?>.*?</q>", "\"", paragraph)
    return paragraph


def add_quote(blockquote, debate_url):
    """Adds a quote (identified by its html element) to the database"""
    try:
        speaker = blockquote.cite.a['title']
        paragraphs = blockquote.find_all("p")
        quote = ""
        for paragraph in paragraphs:
            quote += get_paragraph_text(str(paragraph)) + "\n"
    except TypeError:
        print('Cannot parse quote')


def add_debate(url):
    """Adds the speeches from a debate (identified by its url) to the database"""
    page = urlopen(url)
    page_soup = BeautifulSoup(page, "html.parser")
    blockquotes = page_soup.find_all("blockquote")
    for blockquote in blockquotes:
        add_quote(blockquote, url)

def add_day(day, month, year):
    """Gets the speeches for a given day"""
    url = 'http://hansard.millbanksystems.com/sittings/{}/{}/{}.js'.format(year, month, day)
    res = requests.get(url)
    try:
        obj = json.loads(res.text)
        sections = obj[0]['house_of_commons_sitting']['top_level_sections']
        for section in sections:
            try:
                sec = section['section']
                if 'iraq' in sec['title'].lower():
                    add_debate('http://hansard.millbanksystems.com/commons/{}/{}/{}/{}'
                               .format(year, month, day, sec['slug']))
            except KeyError:
                print('Not a standard section')


    except ValueError:
        print('No data for {} {} {}'.format(day, month, year))

''' add_debate('http://hansard.millbanksystems.com/commons/2003/mar/18/iraq') '''
add_day('18', 'mar', '2003')
