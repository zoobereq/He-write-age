#!/usr/bin/env python
"""A script that takes a text file normalized by normalize.py,
iterates over the lines, capitalizes the first character in each line,
appends a period at the end of it, and concatenates everything into
a continuous string, with each sentence separated by a whitespace"""


import argparse
import string


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
    processed = process(lines)
    one_line = make_continuous(processed)
    write_data(one_line, args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="path to source file")
    parser.add_argument("--output", help="path to output file")
    main(parser.parse_args())
