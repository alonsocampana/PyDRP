# PyDRP
Tool for preprocessing and creating datasets for drug sensitivity, and creating transferable representations across tasks. Contains utils, such as different strategies for featurization of drugs, cell-lines and proteins, and utilities to preprocess the most common public datasets for each tasks, and performing different cross-validation, cold-start settings.  
Example of prediction tasks include:
- Anticancer sensitivity prediction: where cell-lines are represented by different sources of omics data, and drugs are featurized using different strategies.
- Protein-protein interaction prediction: where proteins are represented by their primary sequence, and are later featurized using embeddings obtained from pretrained language models.
- Drug-Target interaction prediction: Proteins are represented by their primary sequences, and drugs are represented by their molecular structure.


`example.ipynb` contains an ilustrative example of its usage.
`ExDTI.ipynb` contains an example for the prediction of binding affinities.
`ExMultiTaskTransferDataset` contains an example where a drug representation is created with the objective of transfer learning.
`ExTabular.ipynb` contains an example where the drugs and cell-lines are converted to tabular features and a baseline performance is obtained.

