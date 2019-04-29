#!/bin/bash
python train.py --feature-type word_pos --domain Q
python train.py --feature-type word_pos --domain N
python train.py --feature-type verbal_stem --domain N
python train.py --feature-type verbal_stem --domain Q
python train.py --feature-type phrase_function --domain N
python train.py --feature-type phrase_function --domain Q
