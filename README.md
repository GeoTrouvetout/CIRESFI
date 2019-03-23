# CIRESFI
see: http://cethefi.org/ciresfi/doku.php?id=en:projet

related to : https://gitlab.univ-nantes.fr/CIRESFI

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

please note that the "artidoc" procedure need the IRONOFF database to be run. please see: http://www.irccyn.ec-nantes.fr/~viardgau/IRONOFF/IRONOFF.html


"ARTIDOC procedure. generate artificial document. the current used procedure is 'randhanddoc'. the 'randhanddoc' arguments are max_size (saize of the generated image), div (let it set to 1), denx (minimum WIDTH of a cell within the randomly created grid, in which we place the IRONOFF patches), deny ((minimum HEIGHT of a cell within the randomly created grid), dropout (density of blank patches. For each grid's cell, the probability to have a blank patches is p=(dropout)/(dropout+3) ), split ('Train', 'Valid', 'Test'), csvdb (RIMES database), csvbg (BACKGROUND database), csvst (STROKES database), snr (Signal-To-Noise Ratio)"





