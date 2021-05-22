#!/bin/bash
echo "Testing Image Runes"
echo ""
ls | parallel --keep-order --tag 'cd {} && rune run "$(find . -name "*.rune")" --capability=image:"$(find . -name "img*")"'