#!/usr/bin/env python
"""A script that normalizes the essays composed by heritage and non-heritage students.
It tokenizes, cleans, and outputs each sentence on a separate line
so that it may be scored for per-character entropy with a ngram langauge model"""

import argparse
import re
import string

import spacy

nlp = spacy.load("pl_core_news_lg")
punctuation = string.punctuation + '…'


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
    with open(path, "w") as sink:
        for sentence in data:
            print(sentence, file=sink)


def make_continuous(data: list) -> string:
    one_line = " ".join(data)
    return one_line


def normalize(data: list) -> list:
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


def main(args: argparse.Namespace) -> None:
    lines = get_data(args.input)
    one_line = make_continuous(lines)
    to_tokenize = nlp(one_line)
    sent_tokenized = [sentence.text for sentence in to_tokenize.sents]
    normalized = normalize(sent_tokenized)
    write_data(normalized, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="path to source file")
    parser.add_argument("--output", help="path to output")
    main(parser.parse_args())
