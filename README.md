 # Deep image prior

In this Folder I have provide *Deep_Image_Prior_Writeup.ipynb* to perform experiments for the paper:

> **Deep Image Prior**

> CVPR 2018

> Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky


[[paper]](https://sites.skoltech.ru/app/data/uploads/sites/25/2018/04/deep_image_prior.pdf) 


# Install

Here is the list of libraries you need to install to execute the code:
- python = 3.6
- [pytorch](http://pytorch.org/) = 0.4
- numpy
- scipy
- matplotlib
- scikit-image
- jupyter
- opencv
- pillow = 6.2.2

All of them can be installed via `conda` (`anaconda`), e.g.
```
conda install jupyter
```


or create an conda env with all dependencies via environment file

```
conda env create -f environment.yml
```

<b>Note:</b> To quickly run the notebook, and visualise the results I have added a flag <code>only_visualize</code>. When set to True, the notebook will use the saved results/images and visualize them. To train the models from scratch, please set this flag as False.

If this flag is set to False, few cells where the best images are handpicked and plotted may not display the best images anymore.