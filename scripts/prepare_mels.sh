#!/usr/bin/env bash

set -e

#DATADIR="/workspace/data/LJ/LJSpeech-1.1"
#FILELISTSDIR="filelists"
#
#TESTLIST="$FILELISTSDIR/ljs_audio_text_test_filelist.txt"
#TRAINLIST="$FILELISTSDIR/ljs_audio_text_train_filelist.txt"
#VALLIST="$FILELISTSDIR/ljs_audio_text_val_filelist.txt"
#
#TESTLIST_MEL="$FILELISTSDIR/ljs_mel_text_test_filelist.txt"
#TRAINLIST_MEL="$FILELISTSDIR/ljs_mel_text_train_filelist.txt"
#VALLIST_MEL="$FILELISTSDIR/ljs_mel_text_val_filelist.txt"
#
#mkdir -p "$DATADIR/mels"
#if [ $(ls $DATADIR/mels | wc -l) -ne 13100 ]; then
#    python preprocess_audio2mel.py --wav-files "$TRAINLIST" --mel-files "$TRAINLIST_MEL"
#    python preprocess_audio2mel.py --wav-files "$TESTLIST" --mel-files "$TESTLIST_MEL"
#    python preprocess_audio2mel.py --wav-files "$VALLIST" --mel-files "$VALLIST_MEL"
#fi


#!/usr/bin/env bash

set -e

DATADIR="/workspace/data/aws/dataset/samantha"
FILELISTSDIR="/workspace/data/aws/dataset/samantha/filenames"

#TESTLIST="$FILELISTSDIR/sm_audio_text_test_filelist.txt"
TRAINLIST="$FILELISTSDIR/train.txt"
#VALLIST="$FILELISTSDIR/sm_audio_text_val_filelist.txt"

#TESTLIST_MEL="$FILELISTSDIR/sm_mel_text_test_filelist.txt"
TRAINLIST_MEL="$FILELISTSDIR/mel_train.txt"
#VALLIST_MEL="$FILELISTSDIR/sm_mel_text_val_filelist.txt"

mkdir -p "$DATADIR/mels"
if [ $(ls $DATADIR/mels | wc -l) -ne 2 ]; then
    python preprocess_audio2mel.py --wav-files "$TRAINLIST" --mel-files "$TRAINLIST_MEL"
fi
