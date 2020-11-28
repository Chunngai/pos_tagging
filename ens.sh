#!/bin/bash

set -e
set -x

TXTS=`ls *txt`
for f in $TXTS
do
	dos2unix $f

	OUT="$f.out"
	echo "$f" > $OUT
	echo -e "\n" >> $OUT
	awk -F $'\t' '{print $2}' $f >> $OUT
done

OUTS=`ls *.out | sort -n`
paste $OUTS > ens_out
