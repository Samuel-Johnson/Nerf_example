# Nerf Example


<a target="_blank" href="https://colab.research.google.com/github/Samuel-Johnson/Nerf_example/blob/branch/colab_example.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

an example implimentation of Neural Radiance Fields (NeRF) described in the paper NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis [paper](https://doi.org/10.48550/arXiv.2003.08934) (Ben Mildenhall et al., 2020).



## Getting Started

### download and install the module

```bash
git clone https://github.com/Samuel-Johnson/Nerf_example.git
cd Nerf_example
python -m pip install .
```

## Running a small demo

download the sample dataset from http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz

for example, you could use wget
```bash
wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz
```

### Run the demo

```bash
python demo.py
```

### Google Colab example

You can also try a demo out in running in google colab. Try setting the runtime to gpu under the Runtime menu at the top of your scene in colab.

<a target="_blank" href="https://colab.research.google.com/github/Samuel-Johnson/Nerf_example/blob/branch/colab_example.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# References
- [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934), Ben Mildenhall et al., 2020
