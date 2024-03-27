# prototypeDL

This repository hosts my Bachelor's thesis for my studies in Artificial Intelligence in JKU. The topic of my Bachelor's thesis is reproducing and testing the functionality and possibilities of PrototypeDL network. This network was hard-coded for the MNIST digit dataset and my goal was to first make it more multi-purpose with being able to apply it on more datasets (both with color information and greyscale), test it on more abstract images (being only able to test the functionality of meaningful prototypes on datasets like MNIST doesn't generally bring a lot of value) and to see whether this can be somehow combined with other explainability methods.

# How to run it?

Example:

```
python CAE_DATASET4.py -i /input/data -c 3 -he 28 -mf test_model -f 0
```

Explanation:

-i: input folder
-c: number of channels
-he: height and width of the images that we want to resize to
-mf: name of the folder that the model outputs will be saved to
-f: 0 or 1, 1 means we want to preprocess the data in the given folder (explanation in the code)
