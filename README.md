# An Introduction to Pattern Recognition

This is just a small program to test different classification algorithms on images. Basically, it works by classifying clusters of pixels belonging to
a certain object. Then, the similarity between a new pixel and the calculated classes is obtained to determine to which of those classes the pixel belongs.

The classification methods ignore stuff such as contours and certain textures in the image. It just calculates the mean of those clusters.

## Requirements

This small program needs python libraries such as **numpy** and **scikit-learn** for classification. Make sure to install necessary dependencies by running:

```bash
$ pip install -r requirements.txt
```

## Usage

The main script loads the graphical interface and the classification libraries.

```bash
$ python run.py
```

### Classification

Use left click to place a new class, and right click to place a pixel that you want to classify.

You may also want to select a classification method from the drop down list.

### Evaluation

Place the classes onto the image. Then, select any evaluation method to display a plot that will calculate the accuracy of the selected algorithm.
