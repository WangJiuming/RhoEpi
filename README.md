# RhoEpi: Precise nucleotide-level RNA modification prediction with deep learning and language model

This is the open source code for RhoEpi.

<details><summary>Citation</summary>

</details>

<details><summary>Table of contents</summary>
  
- [Recent Updates](#New_Updates)
- [Local Environment Setup](#Local_Environment_Setup)
- [Usage](#usage)
  - [Input Arguments](#Arguments)
  - [Input](#Inputs) 
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
python predict.py --help
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path of the input fasta file
  -m MODEL_PATH, --model_path MODEL_PATH
                        Path of the pretrained model
  -g GPU [GPU ...], --gpu GPU [GPU ...]
                        GPU index, default 0, 1
  -l SEQ_LEN, --seq_len SEQ_LEN
                        Length of the input fasta sequence, default 128
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size, default 16
```

### Input <a name="Inputs"></a>

The input file should be a fasta file containing RNA sequences for m6A modification prediction. 

The fasta file can contain any number of RNA sequences. Each RNA sequence should have a length of exactly 128-nt. The sequence can only contain characters A, U, G, C and N.

Example content:
```
>sequence_0
GUGCCAAGUACCCUGCUCCAGGGCGCCUGCAGGAAUAUGGCUCCAUCUUCACGGGCGCCCAGGACCCUGGCCUGCAGAGACGCCCCCGCCACAGGAUCCAGAGCAAGCACCGCCCCCUGGACGAGCGG
>sequence_1
UGAAGAAGCUCUAUGACAGUGAUGUGGCCAAGGUCACCACCCUGAUUUGUCCUGAUAAAGAGAACAAGGCAUAUGUUCGACUUGCUCCUGAUUAUGAUGCUUUCGAUGUUGUAACAAAAUUGGGAUCN
```

### Output <a name="Outputs"></a>

The output file, which is a `.pickle` file of python list storing the nucleotide-level prediction scores, will be saved at `Results/`.

The output of the example input:
```
[[0.02782030776143074, 0.02782030776143074, ..., 0.19342730939388275],    # list of 128 predicted probabilities for the first sequence
 [0.02782030776143074, 0.02782030776143074, ..., 0.02782030776143074]]    # list of 128 predicted probabilities for the second sequence
```


### Example <a name="Examples"></a>

```
python predict.py -i "Data/sample_seq.fasta" -m "Model/epoch=49-val_loss=0.11.ckpt"
```

// ## Citations <a name="Citations"></a>


## License <a name="license"></a>

This source code is licensed under the Apache license found in the `LICENSE` file
in the root directory of this source tree.
