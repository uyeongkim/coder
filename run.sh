#!/bin/bash

python test.py > test_output.txt 2> test_error.txt

python yy_ppo.py > yy_ppo_output.txt 2> yy_ppo_error.txt