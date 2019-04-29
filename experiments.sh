#!/bin/bash
python lvch_2019/train.py --feature-type word_pos --domain Q
python lvch_2019/train.py --feature-type word_pos --domain N
python lvch_2019/train.py --feature-type verbal_stem --domain N
python lvch_2019/train.py --feature-type verbal_stem --domain Q
python lvch_2019/train.py --feature-type phrase_function --domain N
python lvch_2019/train.py --feature-type phrase_function --domain Q
