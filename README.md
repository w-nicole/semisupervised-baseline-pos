
This codebase was first built off a copy of Shijie Wu's crosslingual-nlp repository, found here:

https://github.com/shijie-wu/crosslingual-nlp

# Added text

# Setup

`conda install transformers conllu datasets pandas matplotlib scikit-learn`

Also, as taken from here (5/22/22):

https://pytorch-lightning.readthedocs.io/en/stable/starter/installation.html

`conda install pytorch-lightning -c conda-forge`

Note that pytorch-lightning needs to be 1.4.4 for things to work -- the above command won't install 1.4.4.

If you are using Satori and encounter an error `cannot import name 'create_repo' from 'huggingface_hub'`, you will have to run:

`conda uninstall huggingface_hub` 

`conda install huggingface_hub=0.2.1 transformers` 

# Added text: as of semisupervised-baseline-pos repository

Edits are described in their individual files.

Note that notes on license were not yet added to every file from original repository as many files will be deleted before final state. Ones that are retained are marked as project progresses.

For license information of original repository please see LICENSE file as directed in the leading comments of each file taken from the original. For brevity the entire LICENSE is reproduced only there.

In the final version of this codebase (i.e. not intermediate commits), files that were added later will NOT have an original repository license information comment, but files taken from the original repository will. This is NOT true of intermediate or latest codebase versions, preceding finalization.

# Added text: as of crosslingual-private repository

Commit 2a7b53bfcd1f12c34da410f6799bc18d9adae746

A copy of the crosslingual-nlp repository, with the following general edits:

- Changes to minor hyperparameters to the corresponding paper at https://arxiv.org/abs/1904.09077l.
    - But NOT changes to other processing or details described in the paper that differ from the original code.
- A different environment was used, so the .yml was deleted.
- Simplified `example/surprising-mbert/evaluate.sh` script due to no need for non-POS evaluation.
- A few scripts and files added for convenience of analyzing and executing various runs (`run_multiple_check_variation.sh`, anything with `download` in its name in `src`, `src/scratchwork`)
- Change to `constant.py` to get labels to run with older UD.
- Changes to the way that types are represented in `util.py` to resolve compile errors/possible library inconsistencies.

The above was generated by going through the complete diff as of 3/28/22 (found in the crosslingual-private repository) and changes were described in relevant files.

For the state of the repository before changes, see commit c029e10e909039946a7555ddad87d99e9e0f9fc9 in that repository.

For the state of the repository used for replication numbers before UD change, see commit b1a9af51796f62b9436f2860e44b86891b77be5b in that repository.

# Original text

## Crosslingual NLP

This repo supports various cross-lingual transfer learning & multilingual NLP models. It powers the following papars.

- Mahsa Yarmohammadi*, Shijie Wu*, Marc Marone, Haoran Xu, Seth Ebner, Guanghui Qin, Yunmo Chen, Jialiang Guo, Craig Harman, Kenton Murray, Aaron Steven White, Mark Dredze, and Benjamin Van Durme. [*Everything Is All It Takes: A Multipronged Strategy for Zero-Shot Cross-Lingual Information Extraction*](https://arxiv.org/abs/2109.06798). EMNLP. 2021. ([Experiments Detail](example/data-projection))
- Shijie Wu and Mark Dredze. [*Do Explicit Alignments Robustly Improve Multilingual Encoders?*](https://arxiv.org/abs/2010.02537) EMNLP. 2020. ([Experiments Detail](example/contrastive-alignment))
- Shijie Wu and Mark Dredze. [*Are All Languages Created Equal in Multilingual BERT?*](https://arxiv.org/abs/2005.09093) RepL4NLP. 2020. ([Experiments Detail](example/low-resource-in-mbert))
- Shijie Wu*, Alexis Conneau*, Haoran Li, Luke Zettlemoyer, and Veselin Stoyanov. [*Emerging Cross-lingual Structure in Pretrained Language Models*](https://arxiv.org/abs/1911.01464). ACL. 2020. ([Experiments Detail](example/emerging-crossling-struct))
- Shijie Wu and Mark Dredze. [*Beto, Bentz, Becas: The Surprising Cross-Lingual Effectiveness of BERT*](https://arxiv.org/abs/1904.09077). EMNLP. 2019. ([Experiments Detail](example/surprising-mbert))



## Miscellaneous

- Environment (conda): `environment.yml`
- Pre-commit check: `pre-commit run --all-files`

## License

MIT
