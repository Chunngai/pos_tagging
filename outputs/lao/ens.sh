#!/bin/bash

set -e
set -x

OUT_EXT=".out.tmp"

TXTS=`ls *txt`
for f in $TXTS
do
	dos2unix $f

	OUT="$f$OUT_EXT"
	echo "$f" > $OUT
	echo -e "\n" >> $OUT
	awk -F $'\t' '{print $2}' $f >> $OUT
done

OUTS=`ls *$OUT_EXT | sort -n`
paste $OUTS > ens_out

rm *$OUT_EXT
