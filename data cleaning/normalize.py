#!/usr/bin/env python
"""A script to normalized interview transcripts.
It outputs a single text file with cleaned lines
one sentence per line"""

import argparse
import re
import string
import spacy

fillers = [
    "eh",
    "m",
    "mm",
    "mmm", 
    "ah",
    "ahm",
    "ehm",
    "yy",
    "y",
    "aha",
    "a-ha",
    "aa",
    "e",
    "ee",
    "łyy",
    "ym",
    "yym",
    "ymm",
    "yyym",
    "oh",
    "am",
    "oo",
    "hm",
    "em",
    "emm",
    "eem",
    "yyo",
    "ii",
    "nnn",
    "nn",
    "no",
    "mhm",
    "am",
    "amm",
    "aam",
    "eey",
    "eeyy",
    "mmyy",
    "yhm",
    "ymhm",
    "mmy",
    "yynn",
    "li",
    "cc",
]


nlp = spacy.load("pl_core_news_lg")
punctuation = string.punctuation + '…' +'–' + '’' + "‘"


def get_data(path: str) -> list:
    """reads .txt file into a list of strings"""
    list_of_lines = []
    with open(path, "r") as source:
        for line in source:
            if line == False:
                continue
            else:
                line = line.lstrip().rstrip()
                list_of_lines.append(line)
    return list_of_lines


def write_data(data: list, path: str):
    """writes data line by line into a file"""
    with open(path, "w") as sink:
        for sentence in data:
            print(sentence, file=sink)


def remove_fillers(line: str) -> str:
    """removes filler expresisons"""
    tokens = line.split()
    for word in tokens:
        if word in fillers:
            tokens.remove(word)
    no_fillers = " ".join(word for word in tokens)
    return no_fillers


def pre_tokenization(data: list) -> list:
    """data normalization to be performed before sentence tokenization"""
    cleaned = []
    for line in data:
        # replace ';' and '%' with 'ł'
        add_l = re.sub(r"[;%]", "ł", line)
        # replace the elipses with whitespaces to account for certain types of stutters
        no_elipses = re.sub(r"[…]", " ", add_l)
        # replace two period elipses with whitespace
        two_periods = re.sub(r"\.\.", " ", no_elipses)
        # remove hyphenated stutters
        no_stutters = re.sub(r"\b[a-zA-ZżźćńółęąśŻŹĆĄŚĘŁÓŃ]+-+\W", "", two_periods)
        # remove digits and numbers
        no_numbers = re.sub(r"(?:[+-]|\()?\$?\d+(?:,\d+)*(?:\.\d+)?\)?", "", no_stutters)
        # remove bracketed content
        no_brackets = re.sub(r"\[.*?\]", "", no_numbers)
        # remove content in parentheses
        no_parens = re.sub(r"\(.*?\)", "", no_brackets)
        # remove all duplicate words
        # retain only the first word
        no_duplicates = re.sub(r"\b(\w+)(?:\W+\1\b)+", r"\1", no_parens)
        # append only non-empty strings
        if no_duplicates:
            cleaned.append(no_duplicates)
    return cleaned


def make_continuous(data: list) -> string:
    """joins a list of strings into one long string"""
    one_line = " ".join(data)
    return one_line


def post_tokenization(data: list) -> list:
    """data normaization to be performed after sentence tokenization"""
    cleaned = []
    for sentence in data:
        # casefold and strip
        casefolded = sentence.lower().lstrip().rstrip()
        # standardize quotation marks
        standard_quotes = re.sub(r"[„”“]", '"', casefolded)
        # remove punctuation
        no_polish_punctuation = standard_quotes.translate(str.maketrans("", "", punctuation))
        # remove the hyphens
        no_hyphens = re.sub(r"-", " ", no_polish_punctuation)
        # remove the fillers
        no_fillers = remove_fillers(no_hyphens)
        # remove duplicates left over after the fillers were removed
        # leave only the first word
        no_duplicates = re.sub(r"\b(\w+)(?:\W+\1\b)+", r"\1", no_fillers)
         # remove multiple white spaces
        single_spaces = re.sub(" +", ' ', no_duplicates)
        if single_spaces:
            cleaned.append(single_spaces)
    return cleaned


def main(args: argparse.Namespace) -> None:
    lines = get_data(args.input)
    cleaned = pre_tokenization(lines)
    one_line = make_continuous(cleaned)
    to_tokenize = nlp(one_line)
    sent_tokenized = [sentence.text for sentence in to_tokenize.sents]
    cleaned_again = post_tokenization(sent_tokenized)
    write_data(cleaned_again, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="path to source file")
    parser.add_argument("--output", help="path to output")
    main(parser.parse_args())
