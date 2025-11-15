'''
pip install rdkit #installation
'''

from rdkit import Chem
from rdkit.Chem import Draw


smiles = dataset.smiles[0 ]
print("SMILES:", smiles)

# Convert SMILES to RDKit molecule
molecule = Chem.MolFromSmiles(smiles)

# Visualize molecule
img = Draw.MolToImage(molecule,size=(300, 300))
display(img)
