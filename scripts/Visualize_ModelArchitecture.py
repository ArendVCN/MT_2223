import torch
from Initiate_PSiteDataset import PSiteDataset, re_init_dataset, balance_dataset
from PSite_Models import PSitePredictV4
from torchviz import make_dot
import torchvision
from torchview import draw_graph

path = '/data/home/arendvc/'
saved_model_state = path + 'esm_outputs/ModelV4_ST_CH75_rad30_ESM_E5_0.34014_balanced.pth'

model = PSitePredictV4(input_shape=1280,
                       hidden_units=40,
                       output_shape=2,
                       field_length=61,
                       kernel=3,
                       pad_idx=1,
                       dropout=0.3)
model.load_state_dict(torch.load(saved_model_state))
print(model)
model.eval()

model_graph = draw_graph(model, input_size=(1, 1280, 61), expand_nested=True)
model_graph.visual_graph.render(format='png')
