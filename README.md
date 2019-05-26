# Pytorch Metric Learning

## TODO 

- [ ] Margin contrastive loss, semi-hard 
- [ ] Lifted structured embedding
- [X] TripletLoss
- [ ] Manifold


## Training
    
    cd runs
    bash train-[dataset].sh

## Installation 

    $git clone https://github.com/pedrodiamel/pytorchvision.git
    $cd pytorchvision
    $python setup.py install

### Training visualize

We now support Visdom for real-time loss visualization during training!

To use Visdom in the browser:

    # First install Python server and client 
    pip install visdom
    # Start the server (probably in a screen or tmux)
    python -m visdom.server -env_path runs/visdom/
    # http://localhost:8097/


## Results



## Ref
- https://github.com/kuangliu/pytorch-cifar
- https://github.com/Cadene/pretrained-models.pytorch
- http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html
- https://github.com/vadimkantorov/metriclearningbench



