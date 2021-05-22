#!/bin/bash
echo "Testing Audio Runes"
echo ""
ls | parallel --keep-order --tag 'cd {} && rune run "$(find . -name "*.rune")" --capability=sound:"$(find . -name "aud*")"'