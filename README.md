# RhoEpi: Precise nucleotide-level RNA modification prediction with deep learning and language model

This is the open source code for RhoEpi.

<details><summary>Citation</summary>

</details>

<details><summary>Table of contents</summary>
  
- [Recent Updates](#New_Updates)
- [Local Environment Setup](#Local_Environment_Setup)
- [Usage](#usage)
  - [Input Arguments](#Arguments)
  - [Output](#outputs) 
  - [Example](#Examples)  
- [Citations](#citations)
- [License](#license)
</details>

## Updates <a name="New_Updates"></a>

*** 1 Nov / 2024 ***

Initial commits


## Local Environment Setup <a name="Local_Environment_Setup"></a>

```
conda env create -f RhoEpi_env.yml
```

## Usage <a name="Usage"></a>

### Input Arguments <a name="Arguments"></a>

```
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path of the input fasta file
  -m MODEL_PATH, --model_path MODEL_PATH
                        Path of the pretrained model
  -g GPU [GPU ...], --gpu GPU [GPU ...]
                        GPU index, default 0, 1
  -l SEQ_LEN, --seq_len SEQ_LEN
                        Length of the input fasta sequence
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size, default 16
```


### Output <a name="Outputs"></a>

The output file, which is a `.pickle` file of python list storing the nucleotide-level prediction scores, will be saved at `Results/`.

### Example <a name="Examples"></a>

```
python predict.py --test_path "Data/sample_seq.fasta" --model_path "Model/epoch=49-val_loss=0.11.ckpt"
```

## Citations <a name="Citations"></a>


## License <a name="license"></a>

This source code is licensed under the Apache license found in the `LICENSE` file
in the root directory of this source tree.
