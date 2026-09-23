#!/bin/bash
cd $DIREC # Go inside each folder containing func recordings
# Generate fsf file from template
for i in $FSL_TEMP; do
    sed -e 's@PNMPATH@'$PNMPATHS'@g'\
    		-e 's@OUTDIR@'$OUTDIR'@g' \
            -e 's@NPTS@'"$(fslnvols $DATAPATH)"'@g' \
            -e 's@REPT@'"1.55"'@g' \
    		-e 's@DATAPATH@'$DATAPATH'@g' \
            -e 's@OUTLYN@'$OUTLYN'@g' \
            -e 's@OUTLPATH@'$OUTLPATH'@g' <$i> $FEATOUTPUTNAME.fsf
done