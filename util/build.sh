#!/bin/bash
echo "Building runes - Will not work from projects root"
echo ""

ls | parallel --keep-order --tag 'cd {} && docker run -v `pwd`:`pwd` -w `pwd` hotg/tester /app/rune build'