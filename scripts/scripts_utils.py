import argparse

sources_filenmae_dict = {'drug': 'targets_drug',
                         'AML': 'mutations_AML',
                         'colon': 'mutations_colon',
                         'ovary': 'mutations_ovary',
                         'breast': 'mutations_breast',
                         'yeast': 'sources_yeast'}
terminals_filenmae_dict = {'drug': 'expressions_drug',
                           'AML': 'gene_expression_AML',
                           'colon': 'gene_expression_colon',
                           'ovary': 'gene_expression_ovary',
                           'breast': 'gene_expression_breast',
                           'yeast': 'terminals_yeast'}

model_colors = { 'deep':'goldenrod',
                 'd2d':'royalblue',
                 'vinayagam': 'indianred',
                 'unoriented': 'seagreen',
                 'random':'dimgray'}