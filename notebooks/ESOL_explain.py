import os
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
# %matplotlib inline
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset, MoleculeNet

from models import GCN
from utils.utils import atom_type_count, to_molecule, attribution_visualize, aggregate_edge_directions, attribution_visualize_edge
from captum.attr import Saliency, IntegratedGradients
from torch_geometric.explain import Explainer, PGExplainer, GNNExplainer
from IPython.display import Image, display


config = {
    'dataset': 'ESOL',
    'batch_size': 128,
    'train': False,
    "lr": 0.001,
    "epochs": 100,
    "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    "img_save_dir": "./ouput",
    "explain_topk": 10,
}
device = config['device']
# create output directory
os.makedirs(config['img_save_dir'], exist_ok=True)






def train(model, train_loader, optimizer, loss_fn):
    # Enumerate over the data
    losses = 0
    for batch in train_loader:
        batch.to(device)
        # Reset gradients
        optimizer.zero_grad()
        # Passing the node features and the connection info
        pred = model(batch.x.float(), batch.edge_index, batch.batch)
        # Calculating the loss and gradients
        loss = loss_fn(pred, batch.y)
        loss.backward()
        # Update using the gradients
        optimizer.step()
        losses += loss
    return losses / len(train_loader)


def test(model, test_loader, loss_fn):
    model.eval()
    # regression task
    loss = 0
    for batch in test_loader:
        batch.to(device)
        pred = model(batch.x.float(), batch.edge_index, batch.batch)
        loss += loss_fn(pred, batch.y)
    return loss / len(test_loader)


if __name__ == "__main__":
    dataset = MoleculeNet('MoleculeNet', config['dataset'])
    test_dataset = dataset[:len(dataset) // 10]
    valid_dataset = dataset[len(dataset) // 10:2 * len(dataset) // 10]
    train_dataset = dataset[2 * len(dataset) // 10:]
    test_loader = DataLoader(test_dataset, batch_size=128)
    valid_loader = DataLoader(valid_dataset, batch_size=128)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    print("Dataset type: ", type(dataset))
    print("Dataset features: ", dataset.num_features)
    print("Dataset length: ", dataset.len)
    print("Dataset splits: train, valid, test: ", len(train_dataset), len(valid_dataset), len(test_dataset))
    print("Dataset sample: ", dataset[0])
    print("Sample  nodes: ", dataset[0].num_nodes)
    print("Sample  edges: ", dataset[0].num_edges)

    # count atom types
    atom_types = atom_type_count(dataset)
    print("Atom types: ", atom_types)



    model = GCN()
    print(model)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.7, patience=5,
                                                           min_lr=1e-10)
    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
    if config['train']:
        print("Starting training...")
        train_losses = []
        valid_losses = []
        RMSE = []
        for epoch in range(1, 3001):
            loss = train(model, train_loader, optimizer, loss_fn)
            train_losses.append(loss)
            # valid
            valid_loss = test(model, valid_loader, loss_fn)
            valid_losses.append(valid_loss)
            scheduler.step(valid_loss)
            RMSE.append(np.sqrt(valid_loss.detach().numpy()))

            if epoch % 10 == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch} | Train Loss {loss} | Valid Loss {valid_loss} | LR {lr}")

        train_losses = [float(loss.cpu().detach().numpy()) for loss in train_losses]
        valid_losses = [float(loss.cpu().detach().numpy()) for loss in valid_losses]
        torch.save(model.state_dict(), "gnn_model_esol.pth")

        plt.plot(train_losses, label="Train Loss")

        plt.plot(valid_losses, label="Valid Loss")
        plt.legend()
        plt.grid()
        plt.show()

        plt.plot(RMSE, label="RMSE")
        plt.legend()
        plt.grid()
        plt.show()
    else:
        model.load_state_dict(torch.load(f"gnn_model_{config['dataset'].lower()}.pth"))
        model.eval()


    visualize_index = 0

    data = test_dataset[visualize_index]
    print(data)
    molecule = Chem.MolFromSmiles(dataset[0]["smiles"])
    img_save_path = f"{config['img_save_dir']}/{config['dataset']}/{visualize_index}/original.png"
    os.makedirs(f"{config['img_save_dir']}/{config['dataset']}/{visualize_index}", exist_ok=True)
    attribution_visualize(dataset[0]["smiles"], {}, {}, img_save_path)
    img = Image(filename='{}_atom.png'.format(data["smiles"]))
    display(img)
    print("Original molecule")
    print('-' * 50)

    # from graphlime import GraphLIME
    #
    # # rho controls the strength of the regularization.
    # lime_explainer = GraphLIME(model, hop=2, rho=0.1)
    #
    # # node_idx, data.x, data.edge_index
    # node_idx = 0
    #
    # # Pass in karg as well (batch_index)
    # explanation = lime_explainer.explain_node(node_idx, data.x.float(), data.edge_index,
    #                                           batch_index=torch.zeros(data.x.shape[0], dtype=int))
    #
    # print(explanation)


    def model_forward(edge_mask, data):
        batch = torch.zeros(data.x.shape[0], dtype=int).to(device)
        out = model(data.x.float(), data.edge_index, batch, edge_mask)
        return out
    #
    #
    # def captum_explain(method, data, target=0):
    #     input_mask = torch.ones(data.edge_index.shape[1]).requires_grad_(True).to(device)
    #     if method == 'ig':
    #         ig = IntegratedGradients(model_forward)
    #         mask = ig.attribute(input_mask, target=target,
    #                             additional_forward_args=(data,),
    #                             internal_batch_size=data.edge_index.shape[1])
    #     elif method == 'saliency':
    #         saliency = Saliency(model_forward)
    #         mask = saliency.attribute(input_mask, target=target,
    #                                   additional_forward_args=(data,))
    #     else:
    #         raise Exception('Unknown explanation method')
    #
    #     edge_mask = np.abs(mask.cpu().detach().numpy())
    #     if edge_mask.max() > 0:  # avoid division by zero
    #         edge_mask = edge_mask / edge_mask.max()
    #     return edge_mask
    #
    # for title, method in [('Integrated Gradients', 'ig'), ('Saliency', 'saliency')]:
    #     edge_mask = captum_explain(method, data, target=0)
    #     edge_mask_dict = aggregate_edge_directions(edge_mask, data)
    #     print(f"Explanation for {title}")
    #     img_save_path = f"{config['img_save_dir']}/{config['dataset']}/{visualize_index}/{title}.png"
    #     os.makedirs(f"{config['img_save_dir']}/{config['dataset']}/{visualize_index}", exist_ok=True)
    #     attribution_visualize_edge(data["smiles"], edge_mask_dict, img_save_path)
    #     img = Image(filename=img_save_path)
    #
    #     print('-' * 50)



#     gnnexplainer = Explainer(
#         model=model,
#         algorithm=GNNExplainer(epochs=100),
#         explanation_type='phenomenon',
#         edge_mask_type='object',
#         node_mask_type='object',
#         model_config=dict(
#             mode='regression',
#             task_level='graph',
#             return_type='raw',
#         ),
#         # Include only the top 10 most important edges:
#         threshold_config=dict(threshold_type='topk', value=config['explain_topk']),
#     )
#     batch_index = torch.zeros(data.x.shape[0], dtype=int).to(device)
#     explanation = gnnexplainer(data.x.float(), data.edge_index, target=data.y, batch_index=batch_index)
#     print(data)
#     print(explanation)
#     edge_mask_ = explanation.edge_mask
#     edge_mask_ = edge_mask_.numpy()
#     # edge_mask_ = (edge_mask_ - edge_mask_.min()) / (edge_mask_.max() - edge_mask_.min())
#     edge_mask_ = edge_mask_.tolist()
#     edge_mask_dict = aggregate_edge_directions(edge_mask_, data)
#     node_mask = explanation.node_mask.numpy().tolist()
#
#
#     img_save_path = f"{config['img_save_dir']}/{config['dataset']}/{visualize_index}/gnnexplainer.png"
#     os.makedirs(f"{config['img_save_dir']}/{config['dataset']}/{visualize_index}", exist_ok=True)
#
#     attribution_visualize(data["smiles"], edge_mask_dict, node_mask, img_save_path)
#     # attribution_visualize_edge(data["smiles"], edge_mask_dict, img_save_path)
#     img = Image(filename=img_save_path)
#     display(img)
#
# # node feature
#
    gnnexplainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=100),
        explanation_type='phenomenon',
        node_mask_type='common_attributes',
        model_config=dict(
            mode='regression',
            task_level='graph',
            return_type='raw',
        ),
        # Include only the top 10 most important edges:
        threshold_config=dict(threshold_type='topk', value=config['explain_topk']),
    )

    explanation = gnnexplainer(data.x.float(), data.edge_index, target=data.y, batch_index=batch_index)
    print(explanation)

    # Use GNN explainer to get the node mask for all data: Get average
    node_masks = []
    for data in test_dataset:
        batch_index = torch.zeros(data.x.shape[0], dtype=int).to(device)
        explanation = gnnexplainer(data.x.float(), data.edge_index, target=data.y, batch_index=batch_index)
        node_masks.append(explanation.node_mask.numpy())

