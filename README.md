# DiWTBR-PyTorch
DiWTBR: Dilated Wavelet Transformer for Efficient Megapixel Bokeh Rendering
![](/figs/Figure2.jpg)
# Dependencies
* Python 3.8
* requirements.txt
```bash
pip install requirements.txt #You are now in */DiWTBR_main
```
# Code
We provide code for reproducing results from our paper. You can train our model from scratch, or use our weights to process your images.
# Quickstart (Demo)
You can test our bokeh rendering algorithm with your images. Place your images in ``input`` folder and run the ``test.py``.
If you want to bokeh the images in any folder, please change the folder path on ``parser--test_datadir`` in ``test.py`` to the location you want.We support png and jpg files.

Please download our final checkpoints from [here](https://huggingface.co/Xiaoshi404/DiWTBR/tree/main).When using the specified weight file, please place the weight file in the ``checkpoint`` folder 
and set the ``parser--checkpoint`` in ``test.py``.
# How to train DiWTBR
We trained our model using the FPBNet dataset and EBB! dataset. You can download the dataset to any place you want. Then, change the arguments ``parser--trainset_input`` and ``parser--trainset_output`` in to the place where images are located

Please use ``main.py`` to train the model.We have already set the training parameters for different models. And you need to set it in the ``parser--model_type``.

