# Pytorch Metric Learning (TorchMtc)

## TODO

- [ ] Margin contrastive loss, semi-hard
- [ ] Lifted structured embedding
- [X] TripletLoss
- [ ] Manifold

## Training

```bash
python helper/train.py +configs=<CONFIG>
```

## Installation

```bash
$git clone https://github.com/pedrodiamel/pytorchvision.git
$cd pytorchvision
$python setup.py install
```

### Training visualize

We now support Visdom for real-time loss visualization during training!

To use Visdom in the browser:

```bash
# First install Python server and client
pip install visdom
# Start the server (probably in a screen or tmux)
python -m visdom.server -env_path out/runs/visdom/ -port 6006
# http://localhost:6006/
```

For jupyter notebook

```bash
jupyter notebook --port 8080 --allow-root --ip 0.0.0.0 --no-browser
```

### Docker

```bash
docker build -f "Dockerfile" -t torchmtc:latest .
docker run -ti --privileged --ipc=host --name torchmtc-dev -p 8888:8888 -p 8889:8889 -p localhost:8097:localhost:8097 -v $HOME/.datasets:/.datasets torchtorchmtccls:latest /bin/bash
```

### Dockercompose

```bash
docker-compose up --build -d
docker-compose down
docker exec -it torchmtc-dev /bin/bash
```

## Results

## Reference

- <https://github.com/kuangliu/pytorch-cifar>
- <https://github.com/Cadene/pretrained-models.pytorch>
- <http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html>
- <https://github.com/vadimkantorov/metriclearningbench>
