#!/bin/zsh

function realpath() {
    [[ $1 = /* ]] && echo "$1" || echo "$PWD/${1#./}"
}

function usage()
{
    echo "This script runs Frechet Audio Distance comparing real audio from --real-path and synthesised audio from --synth-path"
    echo ""
}

while [ "$1" != "" ]; do
    PARAM=`echo $1 | awk -F= '{print $1}'`
    VALUE=`echo $1 | awk -F= '{print $2}'`
    case $PARAM in
        -h | --help)
            usage
            exit
            ;;
        --real)
            REAL=$VALUE
            ;;
        --synth)
            SYNTH=$VALUE
            ;;
       --output)
            OUTPUT=$VALUE
            ;;
        *)
            echo "ERROR: unknown parameter \"$PARAM\""
            usage
            exit 1
            ;;
    esac
    shift
done

# !!!       !!!         !!!
# update path to your fad virtual environment
# local:
source "/Users/Marco/Documents/OneDrive - Queen Mary, University of London/PHD/REPOS/hifi-wavegan/google_research_fad/.venv_fad/bin/activate"
# remote:
# source "/homes/mc309/ccwavegan-hifigan-fresh/google_research_fad/.venv_fad/bin/activate"

OUTPUT=`realpath "$OUTPUT"`
REAL=`realpath "$REAL"`
SYNTH=`realpath "$SYNTH"`

# !!!       !!!         !!!
# update path to your google_research_fad repo
# local:
cd "/Users/Marco/Documents/OneDrive - Queen Mary, University of London/PHD/REPOS/hifi-wavegan/google_research_fad"
# remote:
# cd "/homes/mc309/ccwavegan-hifigan-fresh/google_research_fad"

python -m "frechet_audio_distance.create_embeddings_main" --input_files "$REAL" --stats "$OUTPUT/real_stats"
python -m "frechet_audio_distance.create_embeddings_main" --input_files "$SYNTH" --stats "$OUTPUT/synth_stats"

fad=`python -m "frechet_audio_distance.compute_fad" --background_stats "$OUTPUT/real_stats" --test_stats "$OUTPUT/synth_stats"`

echo "$fad"
