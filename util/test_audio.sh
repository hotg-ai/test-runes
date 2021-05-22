#!/bin/bash
echo "Testing Audio Runes"
echo ""
ls | parallel --keep-order --tag 'cd {} && docker run -v `pwd`:`pwd` -w `pwd` hotg/tester /app/rune run "$(find . -name "*.rune")" --capability=sound:"$(find . -name "aud*")"'