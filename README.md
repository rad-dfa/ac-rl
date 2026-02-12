# Automata-Conditioned Reinforcement Learning (AC-RL)

This repo contains a JAX implementation of Automata-Conditioned Reinforcement Learning (AC-RL), a framework for learning multi-task policies where tasks are represented as automata. See `ac_rl/train.py` and `ac_rl/test.py` for example train and test scripts, respectively, both of which has options for running with and without pretrained [RAD Embeddings](https://github.com/rad-dfa/rad-embeddings).

# Citation

Please cite the following papers if you use AC-RL in your work.

```
@inproceedings{DBLP:conf/nips/YalcinkayaLVS24,
  author       = {Beyazit Yalcinkaya and
                  Niklas Lauffer and
                  Marcell Vazquez{-}Chanlatte and
                  Sanjit A. Seshia},
  title        = {Compositional Automata Embeddings for Goal-Conditioned Reinforcement
                  Learning},
  booktitle    = {NeurIPS},
  year         = {2024}
}
```

```
@inproceedings{DBLP:conf/neus/YalcinkayaLVS25,
  author       = {Beyazit Yalcinkaya and
                  Niklas Lauffer and
                  Marcell Vazquez{-}Chanlatte and
                  Sanjit A. Seshia},
  title        = {Provably Correct Automata Embeddings for Optimal Automata-Conditioned
                  Reinforcement Learning},
  booktitle    = {NeuS},
  series       = {Proceedings of Machine Learning Research},
  volume       = {288},
  pages        = {661--675},
  publisher    = {{PMLR}},
  year         = {2025}
}
```
