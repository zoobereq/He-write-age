#!/usr/bin/env python
"""A Supervised ML classifier distinguishing between written output produced by hertiage and non-heritage learners of Polish as a foreign language"""


import math

import numpy as np
import pynini
import spacy
from nltk.tokenize import sent_tokenize
from numpy import ndarray
from sklearn import dummy, feature_extraction, metrics, naive_bayes, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate


class Error(Exception):
    pass


# assigns data to variables
heritage_train = "" # file path for a heritage train corpus
nonheritage_train = "" # file path for a non-heritage train corpus
heritage_test = "heritage_test.txt"
nonheritage_test = "nonheritage_test.txt"
lm = "lm.fst"
nlp = spacy.load("pl_core_news_lg")
M_LN2 = math.log(2)
# prepositions that take LOC for static verbs and ACC for dynamic verbs
prepositions = ["o", "na", "po", "w", "we", "O", "Na", "Po", "W", "We"]
# lemmatized verbs that require a genitive object
verbs = [
    "wymagać",
    "wymagać być",
    "uczyć",
    "ucyć być",
    "nauczyć",
    "nauczyć być",
    "bać",
    "bać być",
    "obawiać",
    "obawiać być",
    "brakować",
    "brakować być",
    "bronić",
    "bronić być",
    "chcieć",
    "chcieć być",
    "napić",
    "napić być",
    "oczekiwać",
    "oczekiwać być",
    "pilnować",
    "pilnować być",
    "potrzebować",
    "potrzebować być",
    "próbować",
    "próbować być",
    "spróbować",
    "spróbować być",
    "pytać",
    "pytać być",
    "szukać",
    "szukać być",
    "słuchać",
    "słuchać być",
    "używać",
    "używać być",
    "doglądać",
    "doglądać być",
    "domagać",
    "domagać być",
    "dotyczyć",
    "dotyczyć być",
    "dotykać",
    "dotykać być",
    "nienawidzić",
    "nienawidzić być",
    "odmawiać",
    "odmawiać być",
    "poszukiwać",
    "poszukiwać być",
    "pozbywać",
    "pozbywać być",
    "pragnąć",
    "pragnąć być",
    "spodziewać",
    "spodziewać być",
    "strzec",
    "strzec być",
    "udzielać",
    "udzielać być",
    "zabraniać",
    "zabraniać być",
    "żałować",
    "żałować być",
    "zapominać",
    "zapominać być",
    "zazdrościć",
    "zazdrościć być",
]


def get_data(path: str) -> list:
    """reads .txt file into a list of strings"""
    list_of_lines = []
    with open(path, "r") as source:
        for line in source:
            line = line.rstrip()
            if line == False:
                continue
            else:
                list_of_lines.append(line)
    return list_of_lines


def get_position(word, sentence):
    """gets the token index in a tokenized sentence"""
    tuples = [(token.text, token.i) for token in sentence]
    for tuple in tuples:
        if tuple[0] == word:
            return tuple[1]


def LOC_post_prep(data: list) -> ndarray:
    """counts the instances of the locative case following bivalent prepositions"""
    bow = []
    for essay_line in data:
        doc = nlp(essay_line)
        counts = []
        for sentence in doc.sents:
            counter = 0
            sentence_string = str(sentence)
            for preposition in prepositions:
                if preposition in sentence_string.split():
                    prep_position = get_position(preposition, sentence)
                    post_prep = doc[prep_position + 1]
                    case = post_prep.morph.get("Case")
                    if case:
                        if case[0] == "Loc":
                            counter += 1
            counts.append(counter)
            summed = sum(counts)
        bow.append(summed)
    bow_np = np.array(bow).astype(float)
    return bow_np


def genitive(data: list) -> ndarray:
    """extracts the counts of direct objects in the genitive case following verbs that take genitive objects """
    bow = []
    for essay_line in data:
        tokenized_line = sent_tokenize(essay_line, language="polish")
        counts = []
        for sentence in tokenized_line:
            tokenized_sentence = nlp(sentence)
            counter = 0
            for token in tokenized_sentence:
                token_str = str(token)
                if token.lemma_ in verbs:
                    verb_position = get_position(token_str, tokenized_sentence)
                    post_verb = tokenized_sentence[verb_position + 1 :]
                    for token in post_verb:
                        case = token.morph.get("Case")
                        if case:
                            if token.dep_ == "obj" and case[0] == "Gen":
                                counter += 1
            counts.append(counter)
            summed = sum(counts)
        bow.append(summed)
    bow_np = np.array(bow).astype(float)
    return bow_np


def negation(data: list) -> ndarray:
    """extracts the counts of direct objects redndered in the genitive case following negated verbs"""
    bow = []
    for essay_line in data:
        tokenized_line = sent_tokenize(essay_line, language="polish")
        counts = []
        for sentence in tokenized_line:
            tokenized_sentence = nlp(sentence)
            counter = 0
            for token in tokenized_sentence:
                token_str = str(token)
                if token.pos_ == "PART" and token.dep_ == "advmod:neg":
                    negation_position = get_position(token_str, tokenized_sentence)
                    post_negation = tokenized_sentence[negation_position + 1 :]
                    for token in post_negation:
                        case = token.morph.get("Case")
                        if case:
                            if token.dep_ == "obj" and case[0] == "Gen":
                                counter += 1
            counts.append(counter)
            summed = sum(counts)
        bow.append(summed)
    bow_np = np.array(bow).astype(float)
    return bow_np


def bits_per_char(string: pynini.Fst, lm: pynini.Fst) -> float:
    """computes bits per char according to the language model (LM FST)"""
    eprops = pynini.ACCEPTOR | pynini.STRING | pynini.UNWEIGHTED
    oprops = string.properties(eprops, True)
    assert eprops == oprops, f"{oprops} != {eprops}"
    lattice = pynini.intersect(string, lm)
    if lattice.start() == pynini.NO_STATE_ID:
        raise Error("Composition failure")
    cost = pynini.shortestdistance(lattice, reverse=True)[lattice.start()]
    bits = float(cost) / M_LN2
    chars = string.num_states() - 1
    return bits / chars


def entropy(data: list, lm: pynini.Fst) -> ndarray:
    """computes per-character entropy for each document"""
    bow = []
    lm = pynini.Fst.read(lm)
    for essay_line in data:
        essay_line = essay_line.rstrip()
        essay_line_fsa = pynini.accep(pynini.escape(essay_line))
        try:
            score = bits_per_char(essay_line_fsa, lm)
            bow.append(score)
        except Error:
            bow.append(0)
    bow_np = np.array(bow).astype(float)
    return bow_np


def cross_validate(data: ndarray, target: ndarray, model) -> float:
    X = data
    Y = np.array(target)
    score = cross_val_score(model, X, Y, scoring="accuracy", cv=10)
    return round(score.mean(), 3)


def main():
    # loads the training data
    heritage_train_input = get_data(heritage_train)  # list of 41 heritage documents
    nonheritage_train_input = get_data(
        nonheritage_train
    )  # list of 36 non-heritage documents
    all_train_input = (
        heritage_train_input + nonheritage_train_input
    )  # list of 77 documents

    heritage_train_labels = [0] * len(heritage_train_input)  # should be 41 "0" labels
    nonheritage_train_labels = [1] * len(
        nonheritage_train_input
    )  # should be 36 "1" labels
    all_train_labels = heritage_train_labels + nonheritage_train_labels  # 77 labels

    # loads the test data
    heritage_test_input = get_data(heritage_test)  # list of 9 heritage documents
    nonheritage_test_input = get_data(
        nonheritage_test
    )  # list of 9 non-heritage documents
    all_test_input = (
        heritage_test_input + nonheritage_test_input
    )  # list of 18 documents

    heritage_test_labels = [0] * len(heritage_test_input)  # should be 9 "0" labels
    nonheritage_test_labels = [1] * len(
        nonheritage_test_input
    )  # should be 9 "1" labels
    all_test_labels = heritage_test_labels + nonheritage_test_labels  # 18 labels

    # Develops a dummy baseline
    vectorizer = feature_extraction.text.CountVectorizer(min_df=3, max_df=0.9)
    D_train = vectorizer.fit_transform(all_train_input)
    D_test = vectorizer.transform(all_test_input)

    dumb = dummy.DummyClassifier()
    dumb = dumb.fit(D_train, all_train_labels)

    dummy_baseline_accuracy = metrics.accuracy_score(
        all_test_labels, dumb.predict(D_test)
    )
    print(
        f"\nDummy baseline accuracy:\t{dummy_baseline_accuracy}"
    )  # this should print 0.5 since there is the same number of heritage and non-heritage essays

    # runs everything on the test data
    multinomial_NB = naive_bayes.MultinomialNB(alpha=1)
    complement_NB = naive_bayes.ComplementNB()
    decision_tree = DecisionTreeClassifier()
    random_forest = RandomForestClassifier()
    svc = svm.SVC(kernel="linear", C=1.0)

    models = [multinomial_NB, complement_NB, decision_tree, random_forest, svc]
    features = [LOC_post_prep, genitive, negation]

    scores_per_model = [
        [
            "Model",
            "LOC after Prepositions",
            "Genitive after Verbs",
            "Genitive of Negation",
            "Per-Character Entropy",
            "All Features Combined",
        ]
    ]

    for model in models:
        feature_acc_scores = [str(model)]
        features_train_list = []
        features_test_list = []
        for feature in features:
            features_train = feature(all_train_input)
            features_train_reshaped = features_train.reshape(-1, 1)
            features_train_list.append(features_train_reshaped)
            features_test = feature(all_test_input)
            features_test_reshaped = features_test.reshape(-1, 1)
            features_test_list.append(features_test_reshaped)
            model.fit(features_train_reshaped, all_train_labels)
            prediction = model.predict(features_test_reshaped)
            accuracy = metrics.accuracy_score(all_test_labels, prediction)
            feature_acc_scores.append(round(accuracy, 3))

        # computes entropy as a separate feature
        entropy_train = entropy(all_train_input, lm)
        entropy_train_reshaped = entropy_train.reshape(-1, 1)
        features_train_list.append(entropy_train_reshaped)
        entropy_test = entropy(all_test_input, lm)
        entropy_test_reshaped = entropy_test.reshape(-1, 1)
        features_test_list.append(entropy_test_reshaped)
        model.fit(entropy_train_reshaped, all_train_labels)
        entropy_prediction = model.predict(entropy_test_reshaped)
        entropy_accuracy = metrics.accuracy_score(all_test_labels, entropy_prediction)
        feature_acc_scores.append(round(entropy_accuracy, 3))

        # computes the score for all features combined
        concatenated_train = np.concatenate(features_train_list, axis=1)
        concatenated_test = np.concatenate(features_test_list, axis=1)
        model.fit(concatenated_train, all_train_labels)
        combined_prediction = model.predict(concatenated_test)
        combined_accuracy = metrics.accuracy_score(all_test_labels, combined_prediction)
        feature_acc_scores.append(round(combined_accuracy, 3))

        # appends everything to the table
        scores_per_model.append(feature_acc_scores)

    # makes and prints the table
    print(f"\nTest Data Accuracy:")
    print(
        f"\n{tabulate(scores_per_model, headers = 'firstrow', tablefmt = 'fancy_grid')}"
    )

    # runs everything on the cross-validated train data
    cv_models = [
        MultinomialNB(),
        ComplementNB(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        SVC(),
    ]
    cv_scores_per_model = [
        [
            "Model",
            "LOC after Prepositions",
            "Genitive after Verbs",
            "Genitive of Negation",
            "Per-Character Entropy",
            "All Features Combined",
        ]
    ]

    for model in cv_models:
        cv_feature_acc_scores = [str(model)]
        for feature in features_train_list:
            cv_accuracy = cross_validate(feature, all_train_labels, model)
            cv_feature_acc_scores.append(cv_accuracy)
        total_cv_accuracy = cross_validate(concatenated_train, all_train_labels, model)
        cv_feature_acc_scores.append(total_cv_accuracy)

        # appends everything to the table
        cv_scores_per_model.append(cv_feature_acc_scores)

    # makes and prints the table
    print(f"\nCross-Validated Training Data Accuracy:")
    print(
        f"\n{tabulate(cv_scores_per_model, headers = 'firstrow', tablefmt = 'fancy_grid')}"
    )


if __name__ == "__main__":
    main()
