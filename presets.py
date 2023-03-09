example_preset = {
    'data':
        {'n_experiments': 5,
         'max_set_size': 500,
         'network_filename': 'H_sapiens.net',
         'directed_interactions_filename': ['KPI'],
         'sources_filename': 'mutations_AML',
         'terminals_filename': 'gene_expression_AML',
         'load_prop_scores': True,
         'save_prop_scores': False,
         'balance_dataset': True,
         'prop_scores_filename': 'AML_KPI_5',
         'random_seed': 0,
         'normalization_method': 'power'},   # standard, power
    'propagation':
        {'alpha': 0.8,
         'eps': 1e-6,
         'n_iterations': 200},
    'model':
        {'feature_extractor_layers': [128, 64, 32],
         'classifier_layers': [64],
         'pulling_func': 'mean',
         'exp_emb_size': 8,
         'feature_extractor_dropout': 0,
         'classifier_dropout': 0,
         'pair_degree_feature': 0,
         'share_source_terminal_weights': False,
         },
    'train':
        {'intermediate_loss_weight': 0.5,
         'bootstrap_sampels': True,
         'intermediate_loss_type': 'BCE', # BCE/FOCAL
         'focal_gamma': 1,
         'train_val_test_split': [0.66, 0.14, 0.2], # sum([train, val, test])=1
         'train_batch_size': 4,
         'test_batch_size': 32,
         'max_num_epochs': 20,
         'eval_interval': 3,
         'learning_rate': 1e-3,
         'max_evals_no_imp': 10,
         'optimizer': 'ADAM'  # ADAM/WADAM
         }}

