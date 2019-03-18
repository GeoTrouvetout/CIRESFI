# coding: utf-8

import sys, getopt

import argparse

import sys
import os
import time

import numpy as np

import pandas as pd
from sklearn.cluster import KMeans
from PIL import Image, ImageEnhance, ImageChops, ImageDraw, ImageFilter

import PIL.ImageOps

import scipy.ndimage
import skimage.morphology
import skimage.measure

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot  as pyplot
import gzip

import theano
import theano.tensor as T

import lasagne
#import nolearn
import csv
import json
import gzip, pickle
# import cv2

from scipy.ndimage import filters

import scipy.misc
import Piff


def trim(im):
#    bg = Image.new('L', im.size)
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    else:
        return im

def enlarge(img, newsize):
    imgEnlarged = Image.new(img.mode, newsize, img.getpixel((0,0)))
    imgEnlarged.paste(img, (0, 0))
    return imgEnlarged

def SpatialSoftmax(x):
    exp_x = T.exp(x - x.max(axis=1, keepdims=True))
    exp_x /= exp_x.sum(axis=1, keepdims=True)
    return exp_x

def build_map_size(input_var=None, sizeX=1024, sizeY=1024):
    l_in = lasagne.layers.InputLayer(shape=(None, 1, sizeX, sizeY), input_var=input_var)
    print(lasagne.layers.get_output_shape(l_in))

    l_ind2 = lasagne.layers.Pool2DLayer(l_in, pool_size=(2,2), mode='max' )
    l_ind2 = lasagne.layers.Upscale2DLayer(l_ind2, scale_factor=2, mode='repeat')
    l_ind4 = lasagne.layers.Pool2DLayer(l_in, pool_size=(4,4), mode='max' )
    l_ind4 = lasagne.layers.Upscale2DLayer(l_ind4, scale_factor=4, mode='repeat')
    l_ind8 = lasagne.layers.Pool2DLayer(l_in, pool_size=(8,8), mode='max' )
    l_ind8 = lasagne.layers.Upscale2DLayer(l_ind8, scale_factor=8, mode='repeat')
    l_ind16 = lasagne.layers.Pool2DLayer(l_in, pool_size=(16,16), mode='max' )
    l_ind16 = lasagne.layers.Upscale2DLayer(l_ind16, scale_factor=16, mode='repeat')

    l_pyr = lasagne.layers.ConcatLayer([l_in, l_ind2, l_ind4, l_ind8, l_ind16], axis=1)
    print(lasagne.layers.get_output_shape(l_pyr))

    l_net = lasagne.layers.Conv2DLayer(l_pyr, 
                                        num_filters=32, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform(), pad='same' )
    l_net = lasagne.layers.MaxPool2DLayer(l_net, pool_size=(2,2) )

    print(lasagne.layers.get_output_shape(l_net))

    l_net = lasagne.layers.Conv2DLayer( l_net, num_filters=32, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotUniform(), pad='same' )
    l_net = lasagne.layers.MaxPool2DLayer(l_net, pool_size=(2,2) )
    print(lasagne.layers.get_output_shape(l_net))

    l_net = lasagne.layers.Conv2DLayer( l_net, num_filters=64, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotUniform(), pad='same' )
    l_net = lasagne.layers.MaxPool2DLayer(l_net, pool_size=(2,2) )
    print(lasagne.layers.get_output_shape(l_net))

    l_net = lasagne.layers.Conv2DLayer( l_net, num_filters=128, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotUniform(), pad='same' )
    l_net = lasagne.layers.MaxPool2DLayer(l_net, pool_size=(2,2) )
    print(lasagne.layers.get_output_shape(l_net))

    l_net = lasagne.layers.Conv2DLayer( l_net, num_filters=256, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotUniform(), pad='same' )
    print(lasagne.layers.get_output_shape(l_net))

    l_net = lasagne.layers.Deconv2DLayer( l_net, 128, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotUniform(), crop='same')
    print(lasagne.layers.get_output_shape(l_net))

    l_net = lasagne.layers.Deconv2DLayer( l_net, 64, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotUniform(), crop='same')
    l_net = lasagne.layers.Upscale2DLayer(l_net, scale_factor=2, mode='repeat')
    print(lasagne.layers.get_output_shape(l_net))

    l_net = lasagne.layers.Deconv2DLayer( l_net, 32, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotUniform(), crop='same')
    l_net = lasagne.layers.Upscale2DLayer(l_net, scale_factor=2, mode='repeat')
    print(lasagne.layers.get_output_shape(l_net))

    l_net = lasagne.layers.Deconv2DLayer( l_net, 16, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotUniform(), crop='same')
    print(lasagne.layers.get_output_shape(l_net))

    #l_net = lasagne.layers.Deconv2DLayer( l_net, 16, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.elu , W=lasagne.init.GlorotUniform(), crop='same')
    #print(lasagne.layers.get_output_shape(l_net))
    l_net = lasagne.layers.Deconv2DLayer( l_net, 8, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotUniform(), crop='same')
    print(lasagne.layers.get_output_shape(l_net))

    l_out = lasagne.layers.Conv2DLayer( l_net, num_filters=3, filter_size=(5,5), nonlinearity=SpatialSoftmax  , W=lasagne.init.GlorotUniform(), pad='same')
    print(lasagne.layers.get_output_shape(l_out))
    l_out = lasagne.layers.Upscale2DLayer(l_out, scale_factor=4, mode='repeat')
    print(lasagne.layers.get_output_shape(l_out))
    return l_out

def convertBoundingBoxesToImage(bboxes):
    img = Image.new('L', (bboxes.shape[1], bboxes.shape[2]))
    draw = ImageDraw.Draw(img)
    for i in np.arange(img.size[0]):
        for j in np.arange(img.size[1]):
            if np.sum(bboxes[5, i, j]) == 1:
                xs, ys, xe, ye = i-bboxes[0,i,j], j-bboxes[1,i,j], i+bboxes[2,i,j], j+bboxes[3,i,j]
                draw.rectangle([xs, ys, xe, ye], fill=255)
    return np.array(img, dtype=np.uint8)

def normalization(mapvalue):
    normmap = (mapvalue-np.min(mapvalue))/(np.max(mapvalue)-np.min(mapvalue))
    return normmap

def standardization(mapvalue):
    standmap = ( mapvalue-np.mean(mapvalue) ) / ( np.std(mapvalue) )
    return standmap

def binarization(probmap, alpha=0.5):
    if np.max(probmap) == np.min(probmap):
        binmap = probmap
    else:
        normmap = (probmap-np.min(probmap))/(np.max(probmap)-np.min(probmap))
        binmap = normmap >= (alpha)
    return np.array(binmap, dtype=bool)

def drawBoundingBoxesFromLabel( img, labelmap, color="blue", thickness=1):
    image = Image.fromarray(img).convert('RGB')
    draw = ImageDraw.Draw(image)
    for l in np.arange(1,np.max(labelmap)+1):
        ccmap = labelmap == l 
        bb = getBoundingBox(ccmap)
        line = (bb[1],bb[0],bb[1],bb[2])
        draw.line(line, fill=color, width=thickness)
        line = (bb[1],bb[0],bb[3],bb[0])
        draw.line(line, fill=color, width=thickness)
        line = (bb[1],bb[2],bb[3],bb[2])
        draw.line(line, fill=color, width=thickness)
        line = (bb[3],bb[0],bb[3],bb[2])
        draw.line(line, fill=color, width=thickness)
#         for t in range(thickness):
#             draw.rectangle( ((bb[1]-t, bb[0]-t), (bb[3]+t, bb[2]+t)) , outline=255, fill=0 )
    return np.array(image)


def drawBoundingBoxesOnImage( img, bboxes, color="blue", thickness=1):
#     image = Image.fromarray(img).convert('RGB')
    image = img
    draw = ImageDraw.Draw(image)
    for l in range(len(bboxes)):
        bb = bboxes[l]
        line = (bb[1],bb[0],bb[1],bb[2])
        draw.line(line, fill=color, width=thickness)
        line = (bb[1],bb[0],bb[3],bb[0])
        draw.line(line, fill=color, width=thickness)
        line = (bb[1],bb[2],bb[3],bb[2])
        draw.line(line, fill=color, width=thickness)
        line = (bb[3],bb[0],bb[3],bb[2])
        draw.line(line, fill=color, width=thickness)
#     return np.array(image)
    return image

def drawBoundingBoxesByLabel( labelmap, thickness=1):
    img = Image.new('L', labelmap.shape)
    draw = ImageDraw.Draw(img)
    for l in np.arange(1,np.max(labelmap)+1):
        ccmap = labelmap == l 
        bb = getBoundingBox(ccmap)
        line = (bb[1],bb[0],bb[1],bb[2])
        draw.line(line, fill=255, width=thickness)
        line = (bb[1],bb[0],bb[3],bb[0])
        draw.line(line, fill=255, width=thickness)
        line = (bb[1],bb[2],bb[3],bb[2])
        draw.line(line, fill=255, width=thickness)
        line = (bb[3],bb[0],bb[3],bb[2])
        draw.line(line, fill=255, width=thickness)
#         for t in range(thickness):
#             draw.rectangle( ((bb[1]-t, bb[0]-t), (bb[3]+t, bb[2]+t)) , outline=255, fill=0 )
    return np.array(img, dtype=np.uint8)



def getBoundingBoxesByLabel(labelmap):
    bboxes = np.array([]).reshape((0,4))
    for l in np.arange(1,np.max(labelmap)+1):
        ccmap = labelmap == l 
        boundingbox = getBoundingBox(ccmap)
#         print(bboxes.shape, boundingbox.shape)
        bboxes = np.concatenate( (bboxes, boundingbox.reshape((1,4))) , axis=0)
    return np.array(bboxes, dtype=np.int)

def getBoundingBox(binmap):
    bbox = np.zeros(4)
    bbox[0] = binmap.shape[0]
    bbox[1] = binmap.shape[1]
    indices = np.argwhere( binmap )
    bbox[0] = np.min(indices[:,0]) 
    bbox[1] = np.min(indices[:,1]) 
    bbox[2] = np.max(indices[:,0]) 
    bbox[3] = np.max(indices[:,1]) 
#     print("---")
    return np.array(bbox, dtype = np.int)
      
def getRectangularity(binmap, bbox):
    As, Ar = 0, 0
#     bbox = np.zeros(4)
#     bbox[0] = binmap.shape[0]
#     bbox[1] = binmap.shape[1]
    As = np.sum(bbox) 
    Ar = np.abs( bbox[2]+1 - bbox[0]-1) * np.abs(bbox[3]+1 - bbox[1]-1)
    return float(As), float(Ar)

def erosion(binmap, n=1, kernel = np.ones((5,5))):
    transimg = binmap
    for i in range(n):
        transimg = skimage.morphology.binary_erosion(transimg, selem=kernel)
    return transimg

def dilation(binmap, n=1, kernel = np.ones((5,5))):
    transimg = binmap
    for i in range(n):
        transimg = skimage.morphology.binary_dilation(transimg, selem=kernel)
    return transimg

def opening(binmap, n=1, kernel = np.ones((5,5))):
    transimg = binmap
    for i in range(n):
        transimg = skimage.morphology.binary_opening(transimg, selem=kernel)
    return transimg

def closing(binmap, n=1, kernel = np.ones((5,5))):
    transimg = binmap
    for i in range(n):
        transimg = skimage.morphology.binary_closing(transimg, selem=kernel)
    return transimg

def binClassificationEvaluation(mask_out, mask_tar):
    imask_out= np.invert(mask_out)
    imask_tar= np.invert(mask_tar)
    TP = np.sum(np.logical_and(mask_out == 1, mask_tar == 1)) 
    # True Negative (TN): we predict a label of 0 (negative), and
    # the true label is 0.
    TN = np.sum(np.logical_and(mask_out == 0, mask_tar == 0))
    # False Positive (FP): we predict a label of 1
    # (positive), but the true label is 0.
    TP = np.sum(np.logical_and(mask_out == 1, mask_tar == 0))
    # False Negative (FN): we predict a label of 0
    # (negative), but the true label is 1.
    FN = np.sum(np.logical_and(mask_out == 0, mask_tar == 1))
    sen = TP/(TP+FN)
    spe = TN/(TN+FP)
    acc = (TP+TN)/(TP+TN+FN+FP)
    pre = TP/(TP+FP)
    f1 = 2*TP/(2*TP+FP+FN)
    mcc =((TP*TN)-(FP*FN)) / np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    return sen, spe, acc, pre, f1, mcc, TP, TN, FP, FN


def classificationStats(out, tar, alpha=0.5):
    mask_out = binarization(normalization(out), alpha)
    mask_tar = binarization(normalization(tar), alpha)

    imask_out= np.invert(mask_out)
    imask_tar= np.invert(mask_tar)
    TP = np.sum(np.logical_and(mask_out == 1, mask_tar == 1)) 
    # True Negative (TN): we predict a label of 0 (negative), and
    # the true label is 0.
    TN = np.sum(np.logical_and(mask_out == 0, mask_tar == 0))
    # False Positive (FP): we predict a label of 1
    # (positive), but the true label is 0.
    FP = np.sum(np.logical_and(mask_out == 1, mask_tar == 0))
    # False Negative (FN): we predict a label of 0
    # (negative), but the true label is 1.
    FN = np.sum(np.logical_and(mask_out == 0, mask_tar == 1))
    
    TP /= 100
    TN /= 100
    FP /= 100
    FN /= 100

    sen = TP/(TP+FN)
    spe = TN/(TN+FP)
    acc = (TP+TN)/(TP+TN+FN+FP)
    pre = TP/(TP+FP)
    f1 = 2*TP/(2*TP+FP+FN)
    mcc =((TP*TN)-(FP*FN)) / np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    return sen, spe, acc, pre, f1, mcc, TP, TN, FP, FN

def imagePadding(img, newsize=(1024,1024)):
    img_pad =Image.new('L', newsize)
    prow, pcol = np.int(np.floor((newsize[0]-img.size[0])/2)), np.int((np.floor(newsize[1]-img.size[1])/2))
    img_pad.paste(img, (prow, pcol))
    return img_pad, prow, pcol


def main(args):

## Set Network ###
    print("Set network")
    input_var = T.tensor4('inputs')
    target_var = T.tensor4('targets')
    class_var = T.ivector('classes')
    seg_var = T.tensor4('segmentations')
    nn = build_map_size(input_var, sizeX=args.insize[0], sizeY=args.insize[1])

    bloadweights = True
    modelnn = args.weights 
    with np.load(modelnn) as f:
        last_param_nn = [f['arr_%d' % i] for i in range(len(f.files))]
    pnn_init = last_param_nn
    lasagne.layers.set_all_param_values(nn, pnn_init)

    lasagne.layers.set_all_param_values(nn, pnn_init)
### Function Definition ###
    print("Fonction Definition")

    out_nn = lasagne.layers.get_output(nn)
    eval_img = theano.function([input_var], out_nn )

    idimg = os.path.basename(args.idimg)

    idcard = idimg # str(idimg[6:8]) + str(idimg[9:13])
    print(idimg, idcard)

    inputs_orig = Image.open( args.idimg ).convert('L')
    #if not args.inzoom == 1:
    #    inputs_orig = inputs_orig.resize( (int(np.floor(inputs_orig.size[0]*args.inzoom)), int(np.floor(inputs_orig.size[1]*args.inzoom))), Image.ANTIALIAS )
    inputs = PIL.ImageOps.invert(inputs_orig)

    oldsize = inputs.size 
    bfit = False
    if inputs.size[0] > args.insize[0]: 
        fsx = args.insize[0]
        bfit = True
    else:
        fsx = inputs.size[0]
    if inputs.size[1] > args.insize[1]: 
        fsy = args.insize[1]
        bfit = True
    else:
        fsy = inputs.size[1]
    fitsize = (fsx, fsy)
    if bfit:
        inputs_fit = inputs.resize( fitsize, Image.ANTIALIAS )
    else:
        inputs_fit = inputs

    newsize = ( int(np.floor(fitsize[0]*args.inzoom)), int( np.floor( args.inzoom*fitsize[1] ) ) )
    #newsize = ( int(np.floor(fitsize[0])), int( np.floor( fitsize[1] ) ) )
    inputs_resized = inputs_fit.resize( newsize , Image.ANTIALIAS )

    inputs_pad, prow, pcol = imagePadding(inputs_resized, newsize=(args.insize[0], args.insize[1]) )

    img = np.array( inputs_pad, dtype=np.float32 )

    restensor = eval_img(img.reshape( (1, 1, args.insize[0], args.insize[1]) ))
    mask_out = np.argmax(restensor[0], axis=0)

    mask_out = mask_out[pcol:pcol+inputs_resized.size[1], prow:prow+inputs_resized.size[0] ] 
    img = img[pcol:pcol+inputs_resized.size[1], prow:prow+inputs_resized.size[0] ] 
    

    bkgd = mask_out == 0 
    num = mask_out == 1
    wor = mask_out == 2 
    print(bkgd.shape, num.shape, wor.shape)
    nimg = np.array( PIL.ImageOps.invert(inputs_resized) , dtype=float)
    image_array = np.array(np.stack(( wor*nimg ,num*nimg, bkgd*nimg), axis=2), dtype=np.uint8)
    if bfit:
        image_array = np.array( Image.fromarray( image_array ).resize(inputs.size, Image.NEAREST) )
    Image.fromarray(image_array, 'RGB').save(args.outpath+idimg+'_outfile.jpg')


    # opening morph transf. to remove small areas 
    num = erosion(num, n=1)
    num = dilation(num, n=1)
    wor = erosion(wor, n=1)
    wor = dilation(wor, n=1)
    img_bb = nimg


    img_num = Image.fromarray(np.array(255*num, dtype=float))
    img_wor = Image.fromarray(np.array(255*wor, dtype=float))
    divsize = 4
    dsize_img_num = ( int( np.floor( img_num.size[0]/divsize ) ), int( np.floor( img_num.size[1]/divsize ) ) )
    # print(img_num, dsize_img_num)
    img_num_d = img_num.resize( dsize_img_num , Image.NEAREST )
    img_wor_d = img_wor.resize( dsize_img_num , Image.NEAREST )
    mask_out_d = Image.fromarray(np.array(mask_out, dtype=float)).resize( dsize_img_num , Image.NEAREST )

    num_d = np.array(img_num_d)
    wor_d = np.array(img_wor_d)


    num_label = skimage.measure.label(num_d, background=0)
    bboxes_num = getBoundingBoxesByLabel(num_label)
    wor_label = skimage.measure.label(wor_d, background=0)
    bboxes_wor = getBoundingBoxesByLabel(wor_label)

    substart_time = time.time()
    minsize = 25
    bboxes_num[ np.logical_and( np.abs(bboxes_num[:,2]-bboxes_num[:,0]) > minsize, np.abs(bboxes_num[:,3]-bboxes_num[:,1]) > minsize ) ]
    bboxes_wor[ np.logical_and( np.abs(bboxes_wor[:,2]-bboxes_wor[:,0]) > minsize, np.abs(bboxes_wor[:,3]-bboxes_wor[:,1]) > minsize ) ]


    bboxes_wor = np.array(divsize * bboxes_wor)# - [pcol,prow,pcol,prow])# + [-5, -5, +5, +5])
    bboxes_num = np.array(divsize * bboxes_num)# - [pcol,prow,pcol,prow])# + [-5, -5, +5, +5])

    img_bb = Image.fromarray(img_bb).convert('RGB')
    img_bboxes = drawBoundingBoxesOnImage( img_bb, bboxes_wor, color="blue", thickness = 10)
    img_bboxes = drawBoundingBoxesOnImage( img_bboxes, bboxes_num, color="green", thickness = 10)


    if bfit:
        img_bboxes = img_bboxes.resize(inputs.size, Image.NEAREST) 
    img_bboxes.save(args.outpath+idimg+'_outbboxes.jpg')



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="'Launch the Structure-Spotting' using pre-trained weights 'nn-weight_structure-spotting.npz'.")
    
    parser.add_argument("-w", "--weights",
            dest="weights",
            type=str,
            default='./',
            help="parameters (weights) of the pre-trained network",)
    parser.add_argument("-r", "--results-path",
            dest="outpath",
            type=str,
            default='./',
            help="output path to save results (npz numpy pickle)",)
    parser.add_argument("-i", "--id-image",
            dest="idimg",
            type=str, 
            help="filename of input image",)
    parser.add_argument("-z", "--zoom-image",
            dest="inzoom",
            type=float, 
            default=1,
            help="zoom to applied to image.",)
    parser.add_argument("-s", "--input-size",
            dest="insize",
            type=int, 
            nargs=2,
            default=(5120, 5120),
            help="size of area in where the input image. Because CNN input need to be square",)
    args = parser.parse_args()
main(args)

