#!/bin/bash

set -e
set -x

TXTS=`ls *txt`
for f in $TXTS
do
	dos2unix $f

	awk -F $'\t' '{print $2}' $f > "$f.out"
done

OUTS=`ls *.out | sort -n`
paste $OUTS > ens_out
