## Her(write)age

A ML classifier to automatically distinguish between written output produced by heritage and non-heritage learners of Polish as a foreign language.

### Motivation
Available at a handful of institutions in the U.S. and Canada, Polish is considered a less-commonly taught language (LCTL).  Similarly to other LCTLs, Polish classes are largely attended by heritage learners, students who “have been raised with a strong cultural connection to a particular language through family interaction” (Van Deusen-Scholl, 2003).  The presence of such students poses a unique set of challenges for curricular design.  In order to develop effective teaching materials instructors need to know the aspects of the target language their heritage students are likely to struggle with, and emphasize those features accordingly.

The goal of this project is twofold.  On the one hand, it hopes to assist instructors in differentiating between heritage students and their non-heritage classmates.  On the other hand, it aims to help instructors isolate the features that are challenging for heritage learners and rank them in the order of salience, such that the features that are most statistically significant in distinguishing between the two groups of learners may be given instructional priority over their less significant counterparts.

### Features
The classifier employs feature sets based on the error types commonly committed by heritage learners of Polish, and absent from non-heritage written output (Wolski-Moskoff, 2019):
- The majority of direct objects in affirmative sentences are expressed in the accusative. Verb negation requires that they be rendered in the genitive case.  Heritage learners tend to ignore that rule, retaining the accusative case in negated declarative constructions.
- Somewhat exceptionally, a handful of verbs require that the direct object always be in the genitive case.  Heritage learners tend to glance over that rule, applying the accusative wherever the genitive is required.
- While the majority of Polish prepositions take a single case, a handful of them can take two or more cases.  There is a group of prepositions that will take either the locative or the accusative case, depending on whether they appear in a static or non-static contexts.  That nuance is largely lost to heritage learners, who tend to overgeneralize the locative case use in contexts that are non-static and therefore requiring an accusative object.

For the purpose of classification, the above feature sets have been reinforced with per-character entropy values computed for each essay.  The entropy calculation is based on a 6-gram character-based (the average Polish word counts 6 characters) language model (`lm.fst`) of the heritage training data.  A model like that can be created with the included script `lm.sh`

### Data
For testing purposes a small corpus of heritage and non-heritage data is provided under `heritage.txt` and `nonheritage.txt` respectively.  Both mini-corpora contain anonymized essays drafted by students of Polish as a foreign languge at a large U.S. university.  Training corpora need to be supplied separately.  The original classifier was trained on documents sourced from the [Corpus of Heritage Language Variation and Change](https://ngn.artsci.utoronto.ca/HLVC/0_0_home.php) (heritage), the [PoLKo - the Polish Learner Corpus](http://utkl.ff.cuni.cz/teitok/polko/index.php?action=home) (non-heritage), as well as a handful oral interviews conducted with heritage speakers of Polish in Chicago.

### Evaluation
The classifier achieves the following accuracy scores for 10-fold cross-validated train data:
| Model                  | LOC after Prepositions | Genitive after Verbs | Genitive of Negation | Per-Character Entropy | All Features Combined |
|------------------------|------------------------|----------------------|----------------------|-----------------------|-----------------------|
| MultinomialNB          | 0.489                  | 0.489                | 0.489                | 0.489                 | 0.668                 |
| ComplementNB           | 0.5                    | 0.5                  | 0.5                  | 0.5                   | 0.668                 |
| DecisionTreeClassifier | 0.853                  | 0.794                | 0.839                | 1                     | 1                     |
| RandomForestClassifier | 0.901                  | 0.744                | 0.826                | 1                     | 1                     |
| SVC                    | 0.864                  | 0.622                | 0.596                | 1                     | 0.864                 |






