CONFIG="libri_probing_example.yaml"
mkdir -p log/$1
cp config/$CONFIG log/$1
sed -i '' s/layer_num\:.*$/layer_num\:\ $3/g log/$1/$CONFIG
python3 main.py --config log/$1/$CONFIG --probing --name $1 --logdir log --ckpdir checkpoint --outdir save --njobs 10 --seed 0 --load $2
