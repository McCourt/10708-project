# Model API

```python
class MyModel(nn.Module):
    def __init__(self, config): # config is a dict
        self.xxx = xxx

    def forward(self, batch):
        x = batch['images']
        z = batch['labels']
        
        output_dict = {}
        generated_images = ...
        output_dict['output_image'] = generated_images
        output_dict['loss'] = loss
        
        # includes output, info for visualization, etc.
        return output_dict
        
    def train(self, dataset):
        # train model
```
