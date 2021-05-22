#!/bin/bash
echo "Building runes - Will not work from projects root"
echo ""

ls | parallel --keep-order --tag 'cd {} && rune build'