#!/bin/sh

cat combined_score_test100_c200.log | grep -E '^Normalized' | perl -pe 's/Normalized Predicted Answer:/NPA:/g;s/Normalized Actual Answer:/NCA:/g;s/^(.+)Predicted.+$/$1/g;s/^(.+)Time:.*/$1/g' > answers.txt