#!/bin/bash
# python test.py --feature-type word_pos --domain Q
# python test.py --feature-type word_pos --domain N
# python test.py --feature-type verbal_stem --domain N
# python test.py --feature-type verbal_stem --domain Q
# python test.py --feature-type phrase_function --domain N
# python test.py --feature-type phrase_function --domain Q
python test.py --feature-type word_number --domain Q
python test.py --feature-type word_number --domain N
python test.py --feature-type clause_type --domain N
python test.py --feature-type clause_type --domain Q

