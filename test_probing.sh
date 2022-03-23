CONFIG="libri_probing_test_example.yaml"
SRC_CONFIG="libri_probing_example.yaml"

[[ $2 == "LibriSpeech" ]] && OUT_DIR="$1" || OUT_DIR="$1_$2";

mkdir -p save/$OUT_DIR
cp config/$CONFIG save/$OUT_DIR
sed -i '' "s/test_path:.*$/test_path:\ \'save\/$2\'/g" "save/$OUT_DIR/$CONFIG";

if [[ $2 == "LibriSpeech" ]];
then
    sed -i '' "s/spec:\ False/spec:\ True/g" "save/$OUT_DIR/$CONFIG";
    sed -i '' "s/wave:\ True/wave:\ True/g" "save/$OUT_DIR/$CONFIG";
fi

sed -i '' "s/src:.*/src:\n\ \ config: \'log\/$1\/$SRC_CONFIG\'\n\ \ ckpt:\ \'checkpoint\/$1\/tts.pth\'/g" "save/$OUT_DIR/$CONFIG"
python3 main.py --config save/$OUT_DIR/$CONFIG --probing --name $OUT_DIR --logdir log --ckpdir checkpoint --outdir save --njobs 0 --seed 0 --test --no-pin
