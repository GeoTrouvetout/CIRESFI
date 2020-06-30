# End-to-end Structures Spotting in Unlabeled Handwritten Documents 

This repository contains the python code to perform structures spotting in handwritten documents. 

This work is based on the paper «Transfer Learning for Structures Spotting in Unlabeled Handwritten Documents using Randomly Generated Documents», International Conference on Pattern Recognition Applications and Methods, 2018. (https://hal.archives-ouvertes.fr/hal-01681114)

In this work, we focus on the localization of structures at the word-level, distinguishing structures *words* from structures *numbers*, in unlabeled handwritten documents. In the related publication, we showed that a transductive transfert learning strategy is able to perform a end-to-end structure spotting. More precisely, we showed that it is possible to construct a coherent map segmentation of word/number/background structures on real documents by using a CNN that was trained on a very large number of synthetical documents (randomly generated).

This work is related to the ANR project CIRESFI
- see: http://cethefi.org/ciresfi/doku.php?id=en:projet
- related to : https://gitlab.univ-nantes.fr/CIRESFI

## Dependencies
- numpy
- pandas
- sklearn
- scipy
- skimage
- matplotlib
- theano
- lasagne
- Piff (Pivot File Format)


## StructSpotting_FCNN_launchOnImage.py
"Launch the Structure-Spotting" using pre-trained weights "nn-weight_structure-spotting.npz". 

Usage:
```
python3 ./StructSpotting_NWB_launchOnImage.py -w nn-weight_structure-spotting.npz -i image.jpg
```

## Learn_NWB-spotting-softmax_ce.py
"Build and Learn a CNN model for structure spotting using artificial document. Artificial document are constructed using the ARTIDOC procedure. "


## artidoc.py

Generate randomly generated synthetic document. The procedure is called "artidoc" and is described in the related publication. 
Please note that the "artidoc" procedure **requires** the IRONOFF database. please see: http://www.irccyn.ec-nantes.fr/~viardgau/IRONOFF/IRONOFF.html

The current used procedure is ```randhanddoc```. 

The ```randhanddoc``` arguments are 

- ```max_size``` : size of the generated image
- ```div``` : NOT USED (let it set to 1)
- ```denx``` : minimum WIDTH of a cell within the randomly created grid, in which we place the IRONOFF patches)
- ```deny``` : minimum HEIGHT of a cell within the randomly created grid
- ```dropout``` : density of blank patches. For each grid's cell, the probability to have a blank patches is $p=(dropout)/(dropout+3)$
- ```split``` : 'Train', 'Valid', 'Test'
- ```csvdb``` : RIMES database
- ```csvbg``` : BACKGROUND database
- ```csvst``` : STROKES database
- ```snr``` : Signal-To-Noise Ratio





