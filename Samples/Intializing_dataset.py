#required modules torch (pip install torch (for installation))
#required modules torch-symmetric (pip install torch-symmetric (for installation))
'''
It is also good to have rdkit for visualization of the molecules
(pip install rdkit)
It uses  smiles for generating the molecule.
'''

from torch_geometric.datasets import QM9

dataset=QM9(root='data/QM9')
print(dataset)
