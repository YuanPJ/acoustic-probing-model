# Initial Layer-to-layer Analysis of End-to-end Automatic Speech Recognition Systems with A Probing-by-reconstruction Model

This project is forked from [End-to-end-ASR-Pytorch](https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch)

See more details in [my paper](#Citation) and master thesis.

## Usage

### Setup

- Clone the project 

    ```git clone git@github.com:YuanPJ/acoustic-probing-model.git```

- Download datasets [LibriSpeech](https://www.openslr.org/12) and [MUSAN](https://www.openslr.org/17/) and put them in save/{dataset_name}

- Install required packages [requirements.txt](/requirements.txt)

    ```pip install -r requirements.txt```
    
### Train ASR

0. Follow [End-to-end-ASR-Pytorch](https://github.com/Alexander-H-Liu/End-to-end-ASR-Pytorch) carefully to train your ASR model.
1. Modify `model` part in configurations [libri_asr_example.yaml](config/libri_asr_example.yaml) 
2. Run [train_asr.sh](train_asr.sh)

    ```bash train_asr.sh your_asr_model_name```

3. ASR model training logs (tensorboard) and configurations will be saved in [log](log/) directory
4. ASR model checkpoints will be saved in [checkpoint](checkpoint/) directory

### Train Probing-by-reconstruction Model

1. Modify `probing` part in configurations [libri_probing_example.yaml](config/libri_probing_example.yaml) 
2. Run [train_probing.sh](train_probing.sh) with trained ASR model checkpoint

    ```bash train_probing.sh your_probing_model_name checkpoint/your_asr_model_name/best_ctc.pth```

3. Probing model training logs (tensorboard) and configurations will be saved in [log](log/) directory
4. Probing model checkpoints will be saved in [checkpoint](checkpoint/) directory

### Test Probing-by-reconstruction Model

- Run [test_probing.sh](test_probing.sh) with configurations [libri_probing_test_example.yaml](config/libri_probing_test_example.yaml)
- Specify the name of dataset. You can augment the original LibriSpeech dataset with noises and put it in save/ directory.

    ```bash test_probing.sh your_probing_model_name dataset_name```

- Reconstructed speech waveform files are in save/your_probing_model_name/ directory

## Citation

```
@INPROCEEDINGS{9054675,  
    author={Li, Chung-Yi and Yuan, Pei-Chieh and Lee, Hung-Yi},  
    booktitle={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},   
    title={What Does a Network Layer Hear? Analyzing Hidden Representations of End-to-End ASR Through Speech Synthesis},   
    year={2020},
    pages={6434-6438},  
    doi={10.1109/ICASSP40776.2020.9054675}
}
```
