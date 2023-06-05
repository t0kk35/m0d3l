# m0d3l
- - -
## Description
Build, train and test Neural Net Models based using Pytorch. This package builds NN models from the artefacts created with the [f3atur3s](https://github.com/t0kk35/f3atur3s) and [eng1n3](https://github.com/t0kk35/eng1n3) packages.

See the [notebooks](https://github.com/t0kk35/m0d3l/tree/main/notebooks) directory for examples.

Example usage
```
# Define the Model
class FirstModel(mp.BinaryClassifier):
    def __init__(self, model_configuration: mp.ModelConfiguration):
        super(FirstModel, self).__init__(model_configuration)
        self.heads = self.create_heads()
        head_size = sum([h.output_size for h in self.heads])
        self.tail = self.create_tail(head_size)

    def forward(self, x: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor,...]:
        o = torch.cat([h(x[i]) for i, h in enumerate(self.heads)], dim=1)
        o = self.tail(o)
        return (o,)

# Create a Model
model = FirstModel(mp.ModelConfiguration.from_tensor_definitions(ti.target_tensor_def))

# Create a trainer.
trainer = mp.Trainer(model, torch.device('cpu'), train_dl, val_dl)
# And an optimizer
optimizer = mp.AdamWOptimizer(model, lr=0.01)
# Run the trainer for 5 epochs
history = trainer.train(5, optimizer)
```

## Requirements
- pandas
- numpy
- numba
- torch
- tqdm
- matplotlib
- scikit-learn