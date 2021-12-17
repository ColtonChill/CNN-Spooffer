# CNN Spooffer
## Instillation
### Cloning this repo
```sh
git clone git@github.com:ColtonChill/CNN-Spooffer.git
cd CNN-Spoffer
```
### Downloading 3-party libraries
You will need to manually install the following libraries to at least these versions:
```
numpy >= 1.19.5
torch >= 1.10.1
torchvision >= 0.10.1
```
Version can be checked with `pip show "_packageName_"`, and downloaded with `pip install "_packageName_"` or updated with `pip install "_packageName_" --upgraded`.

## Running
To use the spooffer, simply run the file `python main.py` from the main project directory. This will begin preforming the gradient ascent algorithm for the specified classes _(see "main.py" line 10)_. To chance the begining class and/or ending class, see `class_table.py` for a list of all possible classes and there index numbers.

To changed these classes, simply modify lines 10 & 11 in `main.py` to the class indices you'd prefer.
```py
init_class_idx = 150  # Sea-lion
target_class_idx = 420  # banjo
```
This will produce a new folder in `output_imgs` labeled the names of the classes you selected, i.e. `sea lion --> banjo/`. Inside will be the initial reference image, as well as the new mutated image which the CNN now evaluates as the specified target class.


### Sources
Tiny ImageNet Tutorial:<br>
[https://towardsdatascience.com/pytorch-ignite-classifying-tiny-imagenet-with-efficientnet-e5b1768e5e8f](https://towardsdatascience.com/pytorch-ignite-classifying-tiny-imagenet-with-efficientnet-e5b1768e5e8f)

Input Image Samples:<br>
[https://github.com/EliSchwartz/imagenet-sample-images](https://github.com/EliSchwartz/imagenet-sample-images)

Preprocessing and Gradient Ascent:<br>
[https://github.com/utkuozbulak/pytorch-cnn-adversarial-attacks](https://github.com/utkuozbulak/pytorch-cnn-adversarial-attacks)

Classification Key:<br>
[https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a)
