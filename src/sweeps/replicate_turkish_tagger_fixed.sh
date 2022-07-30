program: src/train_encoder.py
method: grid
parameters:
    data_dir:
        value: "../../ud-treebanks-v1.4"
    trn_langs:
        value: ["English"]
    val_langs:
        value: ["English", "Turkish"]
    default_save_path:
        value: "./experiments/sweeps/seed/fixed/turkish/fixed/english_only_tagger"
    target_language:
        value: "Turkish"
    group:
        value: "seed/fixed/turkish/english_only_tagger"
    job_type:
        value: "sweep"
    hyperparameter_names:
        value: "seed"
    prep_termination:
        value: no
    batch_size:
        value: 16
    max_epochs:
        value: 3
    weight_decay:
        value: 0.01
    mbert_learning_rate:
        value: 5e-5
    default_learning_rate:
        value: 5e-5
    warmup_portion:
        value: 0.1
    gpus:
        value: 1
    freeze_mbert:
        value: yes
    seed:
        values: [42, 0, 1, 2, 3]
command:
    - ${env}
    - "python3"
    - ${program}
    - ${args}