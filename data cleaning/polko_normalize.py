#!/usr/bin/env python
"""A script to normalize essays extracted from the PoLKo corpus"""

import argparse
import string
import re
import spacy


nlp = spacy.load("pl_core_news_lg")
punctuation = string.punctuation + '…' +'–' + '’' + "‘"


def get_data(path:str) -> str:
    """reads a .txt file"""
    with open(path, 'r') as source:
        lines = source.read()
    return lines


def normalize(data: list) -> list:
    """normalizes a list of sentences"""
    normalized = []
    for sentence in data:
        # casefold and strip
        casefolded = sentence.lower().lstrip().rstrip()
        # standardize quotation marks
        standard_quotes = re.sub(r"[„”“]", '"', casefolded)
        # remove numbers
        no_numbers = re.sub(r"(?:[+-]|\()?\$?\d+(?:,\d+)*(?:\.\d+)?\)?", "", standard_quotes)
        # remove multiple white spaces
        single_spaces = re.sub(" +", ' ', no_numbers)
        # remove punctuation
        no_punctuation = single_spaces.translate(str.maketrans("", "", punctuation))
        if no_punctuation:
            normalized.append(no_punctuation)
    return normalized


def process(data: list) -> list:
    """processes the list of normalized sentences
    by capitalizing the first letter and adding a period at the end"""
    processed_sents = []
    for sentence in data:
        processed = sentence.capitalize() + '.'
        processed_sents.append(processed)
    return processed_sents


def make_continuous(data: list) -> string:
    """joins a list of strings into one long string"""
    one_line = " ".join(data)
    return one_line


def write_data(data: str, path: str):
    """writes out data into a file"""
    with open(path, "w") as sink:
        print(data, file=sink)


def main(args: argparse.Namespace) -> None:
    lines = get_data(args.input)
    to_tokenize = nlp(lines)
    sent_tokenized = [sentence.text for sentence in to_tokenize.sents]
    normalized = normalize(sent_tokenized)
    processed = process(normalized)
    one_line = make_continuous(processed)
    write_data(one_line, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="path to source file")
    parser.add_argument("--output", help="path to output")
    main(parser.parse_args())
