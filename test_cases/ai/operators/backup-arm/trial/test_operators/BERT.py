import torch
from transformers import BertModel, BertTokenizer
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn

model_name = 'bert-base-uncased'
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

def convert_pytorch_to_mxnet(pytorch_model, mxnet_params):
    state_dict = pytorch_model.state_dict()
    #HACK: translate torch matrix to mxnet, it will influence perf result, temporary
    for key, value in state_dict.items():
        mx_value = nd.array(value.numpy())
        mxnet_params[key] = mx_value

mxnet_params = {}

convert_pytorch_to_mxnet(model, mxnet_params)

class MXNetBertModel(nn.Block):
    def __init__(self, **kwargs):
        super(MXNetBertModel, self).__init__(**kwargs)
        self.embeddings = nn.Embedding(30522, 768)  # Example embedding layer
        self.encoder = nn.Sequential()

    def forward(self, inputs):
        # Implement the forward pass
        embeddings = self.embeddings(inputs)
        outputs = self.encoder(embeddings)
        return outputs

mxnet_model = MXNetBertModel()
mxnet_model.initialize(ctx=mx.cpu())

def load_mxnet_params(mxnet_model, mxnet_params):
    for key, value in mxnet_params.items():
        try:
            mxnet_model.collect_params()[key].set_data(value)
        except KeyError:
            print(f"Warning: {key} not found in MXNet model parameters")

load_mxnet_params(mxnet_model, mxnet_params)

text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors='np')
inputs = nd.array(inputs['input_ids'])

outputs = mxnet_model(inputs)
print(outputs)
