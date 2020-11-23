#!/bin/bash

sed $'s/\'/"/g' $1 |
        sed 's/|/\"/' |
        sed 's/é/e/' > "$1.cleaned"

