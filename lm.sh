#!/bin/bash

set -eou pipefail

6-gram-model() {
    farcompilestrings --fst_type=compact --token_type=byte heritage_train.txt train.far
    ngramcount --order=6 --require_symbols=false train.far 6.ct
    ngrammake --method=witten_bell 6.ct lm.fst
}

main() {
    6-gram-model
}

main