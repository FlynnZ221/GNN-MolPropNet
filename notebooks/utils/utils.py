from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw import IPythonConsole
import matplotlib.cm as cm
import matplotlib
from IPython.display import Image
from collections import defaultdict
import matplotlib.pyplot as plt
from rdkit import Chem
from IPython.display import Image, display
import networkx as nx
from torch_geometric.utils import to_networkx
import os


def aggregate_edge_directions(edge_mask, data):
    edge_mask_dict = defaultdict(float)
    for val, u, v in list(zip(edge_mask, *data.edge_index)):
        u, v = u.item(), v.item()
        if u > v:
            u, v = v, u
        edge_mask_dict[(u, v)] += val
    return edge_mask_dict


def attribution_visualize(smiles, edge_mask_dict={}, node_mask=[], img_save_dir='.', cmap_name='Blues'):
    # print the value range of the edge mask
    if edge_mask_dict:
        min_mask_edge = min(edge_mask_dict.values())
        max_mask_edge = max(edge_mask_dict.values())
    else:
        min_mask_edge = 0
        max_mask_edge = 0

    if node_mask:
        node_mask = [x[0] for x in node_mask]
        min_mask_node = min(node_mask)
        max_mask_node = max(node_mask)
    else:
        min_mask_node = 0
        max_mask_node = 0
    mol = Chem.MolFromSmiles(smiles)
    cmap = cm.get_cmap(cmap_name, 10)
    norm_edge = matplotlib.colors.Normalize(vmin=min_mask_edge, vmax=max_mask_edge)
    plt_colors_edges = cm.ScalarMappable(norm=norm_edge, cmap=cmap)

    norm_node = matplotlib.colors.Normalize(vmin=min_mask_node, vmax=max_mask_node)
    plt_colors_nodes = cm.ScalarMappable(norm=norm_node, cmap=cmap)

    highlight_atom_colors = {}
    highlight_bond_colors = {}
    rads = {}

    highlightBonds = defaultdict(float)
    for (u, v), val in edge_mask_dict.items():
        bond = mol.GetBondBetweenAtoms(u, v)
        highlightBonds[bond.GetIdx()] += val

    if node_mask:
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            color = [plt_colors_nodes.to_rgba(node_mask[idx])]
            highlight_atom_colors[idx] = color
            rads[idx] = 0.2

    if edge_mask_dict:
        for bond in mol.GetBonds():
            idx = bond.GetIdx()
            if idx in highlightBonds:
                color = [plt_colors_edges.to_rgba(highlightBonds[idx])]
                highlight_bond_colors[idx] = color


    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
    dos = drawer.drawOptions()
    dos.useBWAtomPalette()
    drawer.DrawMoleculeWithHighlights(mol, smiles, highlight_atom_colors, highlight_bond_colors, rads, {})
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()
    drawer.WriteDrawingText(img_save_dir)
    return png


def attribution_visualize_edge(smiles, edge_mask_dict={}, img_save_dir='.', cmap_name='Blues'):
    # print the value range of the edge mask
    if edge_mask_dict:
        min_mask = min(edge_mask_dict.values())
        max_mask = max(edge_mask_dict.values())
    else:
        min_mask = 0
        max_mask = 0
    mol = Chem.MolFromSmiles(smiles)
    cmap = cm.get_cmap(cmap_name, 10)
    norm = matplotlib.colors.Normalize(vmin=min_mask, vmax=max_mask)
    plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)
    highlight_atom_colors = {}
    highlight_bond_colors = {}
    rads = {}

    highlightBonds = defaultdict(float)
    highlightAtoms = defaultdict(float)
    for (u, v), val in edge_mask_dict.items():
        bond = mol.GetBondBetweenAtoms(u, v)
        highlightBonds[bond.GetIdx()] += val
        highlightAtoms[u] += val
        highlightAtoms[v] += val

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if idx in highlightAtoms:
            color = [plt_colors.to_rgba(highlightAtoms[idx])]
            highlight_atom_colors[idx] = color
            rads[idx] = 0.2

    for bond in mol.GetBonds():
        idx = bond.GetIdx()
        if idx in highlightBonds:
            color = [plt_colors.to_rgba(highlightBonds[idx])]
            highlight_bond_colors[idx] = color
            rads[idx] = 0.2

    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(300, 300)
    dos = drawer.drawOptions()
    dos.useBWAtomPalette()
    drawer.DrawMoleculeWithHighlights(mol, smiles, highlight_atom_colors, highlight_bond_colors, rads, {})
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()
    drawer.WriteDrawingText(img_save_dir)
    return png


def aggregate_edge_directions(edge_mask, data):
    edge_mask_dict = defaultdict(float)
    for val, u, v in list(zip(edge_mask, *data.edge_index)):
        u, v = u.item(), v.item()
        if u > v:
            u, v = v, u
        edge_mask_dict[(u, v)] += val
    return edge_mask_dict



def draw_molecule(g, edge_mask=None, draw_edge_labels=False):
    g = g.copy().to_undirected()
    node_labels = {}
    for u, data in g.nodes(data=True):
        node_labels[u] = data['name']
    pos = nx.planar_layout(g)
    pos = nx.spring_layout(g, pos=pos)
    if edge_mask is None:
        edge_color = 'black'
        widths = None
    else:
        edge_color = [edge_mask[(u, v)] for u, v in g.edges()]
        widths = [x * 10 for x in edge_color]
    nx.draw(g, pos=pos, labels=node_labels, width=widths,
            edge_color=edge_color, edge_cmap=plt.cm.Blues,
            node_color='azure')

    if draw_edge_labels and edge_mask is not None:
        edge_labels = {k: ('%.2f' % v) for k, v in edge_mask.items()}
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels,
                                    font_color='red')
    plt.show()


def to_molecule(data):

    g = to_networkx(data, node_attrs=['x'])
    for u, data in g.nodes(data=True):
        data['name'] = atom_numbers_to_types[data['x'][0]]
        del data['x']
    return g


def atom_type_count(dataset):
    atom_types = set()
    for data in dataset:
        # check smiles
        molecule = Chem.MolFromSmiles(data["smiles"])
        for atom in molecule.GetAtoms():
            atom_types.add(atom.GetSymbol())
    return atom_types

atom_numbers_to_types = {
    17: 'Cl',
    8: 'O',
    35: 'Br',
    6: 'C',
    53: 'I',
    7: 'N',
    15: 'P',
    9: 'F',
    16: 'S',
}

if __name__ == '__main__':

    from torch_geometric.loader import DataLoader
    from torch_geometric.datasets import MoleculeNet
    import warnings

    warnings.filterwarnings("ignore")

    path = '.'
    dataset = MoleculeNet('MoleculeNet', "ESOL")
    test_dataset = dataset[:len(dataset) // 10]
    valid_dataset = dataset[len(dataset) // 10:2 * len(dataset) // 10]
    train_dataset = dataset[2 * len(dataset) // 10:]
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=128)
    train_loader = DataLoader(train_dataset, batch_size=128)

    print("Dataset type: ", type(dataset))
    print("Dataset features: ", dataset.num_features)
    print("Dataset target: ", dataset.num_classes)
    print("Dataset length: ", dataset.len)
    print("Dataset splits: train, valid, test: ", len(train_dataset), len(valid_dataset), len(test_dataset))
    print("Dataset sample: ", dataset[0])
    print("Sample  nodes: ", dataset[0].num_nodes)
    print("Sample  edges: ", dataset[0].num_edges)

    # count atom types
    atom_types = set()
    for data in dataset:
        # check smiles
        molecule = Chem.MolFromSmiles(data["smiles"])
        for atom in molecule.GetAtoms():
            atom_types.add(atom.GetSymbol())
    print("Atom types: ", atom_types)
    data = dataset[0]
    print(data)
    molecule = Chem.MolFromSmiles(dataset[0]["smiles"])
    atom_attribution_visualize(dataset[0]["smiles"])
    img = Image(filename='{}_atom.png'.format(data["smiles"]))
    display(img)
    print("Original molecule")
    print('-' * 50)

    for title, method in [('Integrated Gradients', 'ig'), ('Saliency', 'saliency')]:
        edge_mask = explain(method, data, target=0)
        edge_mask_dict = aggregate_edge_directions(edge_mask, data)
        print(f"Explanation for {title}")
        atom_attribution_visualize(data["smiles"], edge_mask_dict)
        img = Image(filename='{}_atom.png'.format(data["smiles"]))
        display(img)
        print('-' * 50)