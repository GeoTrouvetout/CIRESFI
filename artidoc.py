
# coding: utf-8

import sys, getopt

import argparse

import sys
import os
import time

import numpy as np

import pandas as pd
from sklearn.cluster import KMeans
from PIL import Image, ImageChops, ImageDraw
import PIL.ImageOps

import scipy.misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot  as pyplot
import gzip

import csv
import gzip, pickle


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


def randhanddoc(max_size=(1024,1024), div=1, denx=100, deny=100, dropout=1,  split='All', csvdb="dbImages.csv", csvbg="dbBackground.csv", csvst='dbStrokes.csv', snr=10):
 
    csvDB = pd.read_csv(csvdb, delimiter=';')
    dircsvDB = os.path.dirname(csvdb)
    csvDBfond = pd.read_csv(csvbg, delimiter=';')
    dircsvDBfond = os.path.dirname(csvbg)
    csvDBstrokes = pd.read_csv(csvst, delimiter=';')
    dircsvDBstrokes = os.path.dirname(csvst)        
    def generationPatch():
    
        ListDigit=('0','1','2','3','4','5','6','7','8','9') 
        ListDigitExt=('0','1','2','3','4','5','6','7','8','9', 'o','O','i','I', 'z', 'Z', 's', 'S' ,'g', 'G') 
        ListChar=('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'euro') 
        ListCharDec=('a', 'b', 'c', 'd', 'e', 'f', 'h', 'k', 'm', 'n', 'p', 'r', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'H', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'T', 'U', 'V', 'W', 'X', 'Y', 'euro') 
        
        if split == 'All':
            probSplit = np.ones(len(csvDB))
        else:
            probSplit = np.where(csvDB['Cohort'] == split, 1, 0)

        rType = np.random.choice(np.arange(1-dropout, 4, 1),1)
        
        patchType = 'Bckgd'
        bTypebck = True
        
        img_trim1 = Image.new('L', (24,24))
#         if rType <= 1-dropout  :

        if rType == 1:
            bTypebck = False
            bTypeNum = True
            bDigitPresence = True
            patchType = 'Digit' 
            rnbDigit = np.random.choice(np.arange(1,6),1)
            if rnbDigit > 1:
                patchType = 'Number'
            listSearch = ListDigit 
            prob = np.where( np.array(csvDB['Type']) == 'Digit' , 1, 0 ) 
#             prob = np.where( np.searchsorted( listSearch , np.array(csvDB['Content']) ) < len(listSearch) , 1, 0 ) * probSplit
            img =Image.new('L', (1000,1000))
            px, py = 300, 300
            prev_rand_px, prev_rand_py = 0, 0 
            for i in np.arange(rnbDigit):    
                n = np.random.choice(np.arange(0,len(prob)), 1, p=prob/np.sum(prob))
                filen = np.array(csvDB['PathAbs'][n])[0]
                Contentn = np.array(csvDB['Content'][n])[0]
#                 print(patchType, Contentn.encode('utf-8'))
                Typen = np.array(csvDB['Type'][n])[0]  
                patch = Image.open(dircsvDB+ "/"+str(filen)).convert('L')
                patch = PIL.ImageOps.invert(patch)
                patch_trim = trim(patch)
                img_patch =Image.new('L', (1000,1000))
                img_patch.paste(patch_trim, (px, py))
                img = ImageChops.add(img, img_patch)
                rand_px = np.random.choice( np.arange(-10,15), 1 )
                rand_py = np.random.choice( np.arange(-10,15), 1 )
                px = px + patch_trim.size[0] + rand_px
                py = py + rand_py
            img_trim1 = trim(img)

        if rType == 20: # Isolated Char removed (to set put 'rType == 2')
            bTypebck = False
            bTypeChar = True
            patchType = 'Char' 
            listSearch = ListChar
            prob = np.where( np.array(csvDB['Type']) == 'Character' , 1, 0 ) 
            n = np.random.choice(np.arange(0,len(prob)), 1, p=prob/np.sum(prob))
            filen = np.array(csvDB['PathAbs'][n])[0]
            Contentn = np.array(csvDB['Content'][n])[0]
#             print(patchType, Contentn.encode('utf-8'))
            if Contentn in ListDigit:
                patchType = 'Digit'
            Typen = np.array(csvDB['Type'][n])[0]
            img = Image.open(dircsvDB+ "/" + str(filen)).convert('L')
            img = PIL.ImageOps.invert(img)
            img_trim1 = trim(img)

        if rType == 3:
            bTypebck = False
            bTypeWord = True
            patchType = 'Word' 
            listSearch = ('FrenchWord', 'EnglishWord')
            prob = np.where( np.array(csvDB['Type']) == 'FrenchWord' , 1, 0 ) 
            prob = np.where( np.array(csvDB['Type']) == 'EnglishWord' , 1, 0 ) + prob
            prob = prob * probSplit
            n = np.random.choice(np.arange(0,len(prob)), 1, p=prob/np.sum(prob))
            filen = np.array(csvDB['PathAbs'][n])[0]
            Contentn = np.array(csvDB['Content'][n])[0]
            Typen = np.array(csvDB['Type'][n])[0]    
            img = Image.open(dircsvDB+ "/" + str(filen)).convert('L')
            img = PIL.ImageOps.invert(img)
            img_trim1 = trim(img)

        return img_trim1, patchType
     
    #max_size = (512, 512)
    bDigitPresence = False
    rvm = np.random.randint(20)
    arrgenRanImg = np.random.randint(50, size=max_size[0]*max_size[1]).reshape(max_size)
    genRanImg =Image.new('L', max_size)
    
    pos_available = np.ones(max_size)
    ind_available = np.arange(max_size[0]*max_size[1]).reshape(max_size)
    div_xaxis = np.random.randint(1,max_size[0]/denx)
    div_yaxis = np.random.randint(1,max_size[1]/deny)
    rsx = int(max_size[0]/div)
    rsy = int(max_size[1]/div)
    bounding_boxes= np.zeros((5,rsx,rsy))
    mask_structures= np.zeros((5,rsx,rsy))
    mask_NWB= np.zeros((5,rsx,rsy))
    prob_structures= np.zeros((5,rsx,rsy))
    prob_type= np.zeros((5,rsx,rsy))
    cwh_boxes= np.zeros((3,rsx,rsy))
    bounding_boxes[0,:,:] = 1
    mask_structures[0,:,:] = 1
    mask_NWB[0,:,:] = 1
    prob_structures[0,:,:] = 1
    bTypeNum = False
    denDigit = 4086/61292
    denChar = 25860/61292
    denWord = 31346/61292
    couDig = 0 
    couCha = 0 
    couWor = 0 
    for w in np.arange(div_xaxis):
        for h in np.arange(div_yaxis):
            bTypeNum = False
            bTypeChar = False
            bTypeWord = False
            bTypebck = False
            
            if split == 'All':
                probSplit = np.ones(len(csvDB))
            else:
                probSplit = np.where(csvDB['Cohort'] == split, 1, 0)
                    
            img_trim, patchType = generationPatch()

            rRescaleFactor_x = np.random.choice(np.arange(0.5,1.55,0.05), 1)
            rRescaleFactor_y = np.random.choice(np.arange(0.5,1.55,0.05), 1)
            img_trim = img_trim.resize( ( int(np.round(img_trim.size[0]*rRescaleFactor_x)), int(np.round(img_trim.size[1]*rRescaleFactor_y)))  , Image.ANTIALIAS )
            min_xy = (int(w*max_size[0]/div_xaxis), int(h*max_size[1]/div_yaxis) )
            max_xy = (int(((w+1)*max_size[0]/div_xaxis)), int((h+1)*(max_size[1]/div_yaxis)))
            size_wind=(max_xy[0] - min_xy[0] ,max_xy[1]-min_xy[1] )

            if img_trim.size[0] > max_xy[0]-min_xy[0] or img_trim.size[1] > max_xy[1]-min_xy[1]:
                if patchType in ('Digit', 'Char', 'Bckgd'):
                    #img_trim = img_trim.resize( size_wind , Image.ANTIALIAS )
                    img_trim.thumbnail(size_wind)
                else:
                    if np.random.randint(0,500) > 1 :
                        img_trim.thumbnail(size_wind)
                    else:
                        img_trim = img_trim.resize( size_wind , Image.ANTIALIAS )

            
            umax_x, umax_y = int((max_xy[0] - img_trim.size[0] )), int((max_xy[1] - img_trim.size[1]))
            choice_xy = (umax_x, umax_y)
            px, py = min_xy[0] , min_xy[1]
            if not umax_x <= min_xy[0]:
                px = np.random.randint(min_xy[0], umax_x)
            if not umax_y <= min_xy[1]:
                py = np.random.randint(min_xy[1], umax_y)
            genRanImg.paste(img_trim, (px, py))
            six, siy = img_trim.size[0], img_trim.size[1]
            sx, sy = img_trim.size[0]/div, img_trim.size[1]/div
            pxd, pyd = np.floor(px/div).astype(int) , np.floor(py/div).astype(int)
            scx, ecx, scy, ecy = np.floor(sx/2).astype(int), np.ceil(sx/2).astype(int), np.floor(sy/2).astype(int), np.ceil(sy/2).astype(int)
            cx, cy = pxd+scx, pyd+scy
            
            
            # Digit is Char
            # Digit is Number
            # Char is Word
            # Word is not 
            # Word is not Char 
            # Digit in Number
            # Word in Char
            # Number not in Char
            # Number not in Word
            # Number not in Digit
            # Word not in Number
            # Word not in Char
            #
            if patchType == 'Digit':
                mask_structures[0, pyd:pyd+siy,pxd:pxd+six] = 0 
                mask_structures[1, pyd:pyd+siy,pxd:pxd+six] = 1
                mask_structures[2, pyd:pyd+siy,pxd:pxd+six] = 0
                mask_structures[3, pyd:pyd+siy,pxd:pxd+six] = 0
                mask_structures[4, pyd:pyd+siy,pxd:pxd+six] = 0
                cx, cy = int(pxd+np.round(six/2)) , int(pyd+np.round(siy/2))
                cwh_boxes[0, cx, cy] = 1
                cwh_boxes[1, cx, cy] = six / 2
                cwh_boxes[2, cx, cy] = siy / 2    
                
            if patchType == 'Number':
                mask_structures[0, pyd:pyd+siy,pxd:pxd+six] = 0 
                mask_structures[1, pyd:pyd+siy,pxd:pxd+six] = 0 
                mask_structures[2, pyd:pyd+siy,pxd:pxd+six] = 1 
                mask_structures[3, pyd:pyd+siy,pxd:pxd+six] = 0 
                mask_structures[4, pyd:pyd+siy,pxd:pxd+six] = 0 
                cx, cy = int(pxd+np.round(six/2)) , int(pyd+np.round(siy/2))
                cwh_boxes[0, cx, cy] = 1
                cwh_boxes[1, cx, cy] = six / 2
                cwh_boxes[2, cx, cy] = siy / 2

            if patchType == 'Char':
                mask_structures[0, pyd:pyd+siy,pxd:pxd+six] = 0
                mask_structures[1, pyd:pyd+siy,pxd:pxd+six] = 0
                mask_structures[2, pyd:pyd+siy,pxd:pxd+six] = 0
                mask_structures[3, pyd:pyd+siy,pxd:pxd+six] = 1
                mask_structures[4, pyd:pyd+siy,pxd:pxd+six] = 0
                cx, cy = int(pxd+np.round(six/2)) , int(pyd+np.round(siy/2))
                cwh_boxes[0, cx, cy] = 1
                cwh_boxes[1, cx, cy] = six / 2
                cwh_boxes[2, cx, cy] = siy / 2

            if patchType == 'Word':
                mask_structures[0, pyd:pyd+siy,pxd:pxd+six] = 0
                mask_structures[1, pyd:pyd+siy,pxd:pxd+six] = 0
                mask_structures[2, pyd:pyd+siy,pxd:pxd+six] = 0
                mask_structures[3, pyd:pyd+siy,pxd:pxd+six] = 0
                mask_structures[4, pyd:pyd+siy,pxd:pxd+six] = 1
                cx, cy = int(pxd+np.round(six/2)) , int(pyd+np.round(siy/2))
                cwh_boxes[0, cx, cy] = 1
                cwh_boxes[1, cx, cy] = six / 2
                cwh_boxes[2, cx, cy] = siy / 2


            if patchType == 'Bckgd':
                mask_structures[0, pyd:pyd+siy,pxd:pxd+six] = 1
                mask_structures[1, pyd:pyd+siy,pxd:pxd+six] = 0
                mask_structures[2, pyd:pyd+siy,pxd:pxd+six] = 0
                mask_structures[3, pyd:pyd+siy,pxd:pxd+six] = 0
                mask_structures[4, pyd:pyd+siy,pxd:pxd+six] = 0

    
    mask_structures[2,:,:] += mask_structures[1,:,:]
    mask_structures[4,:,:] += mask_structures[3,:,:]
                
    nb_pixBg =  (np.sum(mask_structures[0,:,:]))
    nb_pixDi =  (np.sum(mask_structures[1,:,:]))
    nb_pixNu =  (np.sum(mask_structures[2,:,:]))
    nb_pixCh =  (np.sum(mask_structures[3,:,:]))
    nb_pixWo =  (np.sum(mask_structures[4,:,:]))

    p_pixBg = (1 + nb_pixBg) / (1 + nb_pixNu + nb_pixWo + nb_pixBg)
    p_pixNu = (1 + nb_pixNu) / (1 + nb_pixBg + nb_pixWo + nb_pixNu)    
    p_pixWo = (1 + nb_pixWo) / (1 + nb_pixBg + nb_pixNu + nb_pixWo)
    p_pixSi = (nb_pixNu + nb_pixWo) / (nb_pixBg + nb_pixNu + nb_pixWo)
#     p_pixBg =  (np.sum((mask_structures)[0,:,:]))/(max_size[0]*max_size[1])
    p_pixDi =  (1+np.sum((mask_structures)[1,:,:]))/(1+max_size[0]*max_size[1])
#     p_pixNu =  (np.sum((mask_structures)[2,:,:]))/(max_size[0]*max_size[1])
    p_pixCh =  (1+np.sum((mask_structures)[3,:,:]))/(1+max_size[0]*max_size[1])
#     p_pixWo =  (np.sum((mask_structures)[4,:,:]))/(max_size[0]*max_size[1])
    p_pixStructure =  (1+np.sum((cwh_boxes)[0,:,:]))/(1+max_size[0]*max_size[1])
    
    cwh_boxes[0,:,:] *= 1 - p_pixStructure
    prob_type[0,:,:] = 1-p_pixBg # np.where(mask_structures[0,:,:] == 1, 1-p_pixBg, p_pixBg)
    prob_type[1,:,:] = 1-p_pixDi #np.where(mask_structures[1,:,:] == 1, 1-p_pixDi, p_pixDi)
    prob_type[2,:,:] = 1-p_pixNu #np.where(mask_structures[2,:,:] == 1, 1-p_pixNu, p_pixNu)
    prob_type[3,:,:] = 1-p_pixCh #np.where(mask_structures[3,:,:] == 1, 1-p_pixCh, p_pixCh)
    prob_type[4,:,:] = 1-p_pixWo #np.where(mask_structures[4,:,:] == 1, 1-p_pixWo, p_pixWo)

    prob_structures[0,:,:] = np.where(mask_structures[0,:,:] == 1, p_pixBg, 1-p_pixBg)
    prob_structures[1,:,:] = np.where(mask_structures[1,:,:] == 1, p_pixDi, 1-p_pixDi)
    prob_structures[2,:,:] = np.where(mask_structures[2,:,:] == 1, p_pixNu, 1-p_pixNu)
    prob_structures[3,:,:] = np.where(mask_structures[3,:,:] == 1, p_pixCh, 1-p_pixCh)
    prob_structures[4,:,:] = np.where(mask_structures[4,:,:] == 1, p_pixWo, 1-p_pixWo)

    
    n = np.random.choice(np.arange(0,len(csvDBfond)), 1)
    n=n[0]
    backgdn = csvDBfond['Path'][n]
    bckgd = Image.open( dircsvDBfond+ "/" + backgdn).convert('L')
    bckgd = PIL.ImageOps.invert(bckgd)
    if np.random.randint(0,2) < 1 :
        bckgd.rotate(90)
    rCBckgdx = np.random.choice(np.arange(100,bckgd.size[0]-max_size[0]), 1)
    rCBckgdy = np.random.choice(np.arange(100,bckgd.size[1]-max_size[1]), 1)
#     print(bckgd.size, (rCBckgdx[0], rCBckgdy[0], (rCBckgdx+max_size[0])[0], (rCBckgdy+max_size[1])[0]))
    bckgd = bckgd.crop((rCBckgdx[0], rCBckgdy[0], (rCBckgdx+max_size[0])[0], (rCBckgdy+max_size[1])[0]))
    #bkdout = np.array(bckgd, dtype=np.uint8)
    #genRanImg = ImageChops.add(genRanImg, genRanImg)
    
    nStrokes = np.random.choice(np.arange(div_xaxis*div_yaxis), 1)
    for istr in np.arange(nStrokes):
        
        n = np.random.choice(np.arange(0,len(csvDBstrokes)), 1)
        n = n[0]
        stroken = csvDBstrokes['Path'][n]
        stroke = Image.open(dircsvDBstrokes+ "/"+stroken).convert('L')
        stroke = PIL.ImageOps.invert(stroke)
        arrstroke = np.array(stroke)
        stroke = Image.fromarray ( (arrstroke > 0.1*(np.max(arrstroke)-np.min(arrstroke))) * arrstroke )
        stroke_rx, stroke_ry = np.random.choice(np.arange(0.25,2.25,0.25),1)[0], np.random.choice(np.arange(0.25,2.25,0.25),1)[0] 
        
        stroke = stroke.resize( (int(np.round(stroke_rx*stroke.size[0])), int(np.round(stroke_ry*stroke.size[1]))) , Image.ANTIALIAS )
        
        if np.random.randint(0,4) < 1 :
            stroke.rotate(np.random.randint(0,360))
        
        stroke = Image.fromarray( np.random.choice(np.arange(0,0.5,0.1) ,1) *  np.array(stroke) * (np.array(stroke)> 0.5*np.max(np.array(stroke)) ) )
        px, py = np.random.randint( max_size[0] ),  np.random.randint( max_size[1] )
        img_stroke = Image.new('L', max_size)
        img_stroke.paste(stroke, (px, py))
         
        bckgd = ImageChops.add(bckgd, img_stroke)
#         bckgd = Image.blend(bckgd, img_stroke, (np.random.randint(11)/10) )
    
#     imartidoc = Image.blend( bckgd, genRanImg, 0.5) # ((5+np.random.randint(3))/10) )
    imartidoc = ImageChops.add(genRanImg, bckgd)
    artidoc = np.array(imartidoc, dtype=np.uint8)
    sigma_noise = np.var(artidoc) / np.power(10, snr/10)
 
    noise = (np.random.normal(0,sigma_noise, max_size[0]*max_size[1])).reshape(max_size)

    artidoc = artidoc + noise

    genRanImg.close()
    img_trim.close()

    return artidoc, mask_structures, prob_structures, bDigitPresence, bckgd

   

def main(args):


    ### Set Network ###
    sizeInput = args.imgsize
    nimg = 0
    for batch in np.arange( args.nbimg ):
        idartidoc = '%02d'%batch
        snrn = np.random.choice(np.arange(10,100,10))
    
        inpt, maskm, probm, bDigPresTrain, bckgd = randhanddoc(max_size=sizeInput, 
                                                                        div=1, 
                                                                        denx=args.genparam[0],
                                                                        deny=args.genparam[1], 
                                                                        dropout=args.genparam[2], 
                                                                        split=args.cohort,
                                                                        csvdb=args.csvdb, 
                                                                        csvbg=args.csvbg,
                                                                        csvst=args.csvst,
                                                                        snr = snrn)
        
        mask = np.array(maskm, dtype=np.uint8)
        img = np.max(inpt) - inpt
        bkgd = mask[0,:,:]
        dig = mask[1,:,:]
        num = mask[2,:,:]
        char = mask[3,:,:]
        word = mask[4,:,:]

        
#             imgmask = np.argmax(mask, axis=0)
        scipy.misc.toimage(img).save(args.outpath+'artidoc_'+args.cohort+"_"+idartidoc+'.jpg')
        scipy.misc.toimage(bkgd).save(args.outpath+'artidoc_'+args.cohort+"_"+idartidoc+'_bkgd'+'.jpg')
        scipy.misc.toimage(dig).save(args.outpath+'artidoc_'+args.cohort+"_"+idartidoc+'_dig'+'.jpg')
        scipy.misc.toimage(num).save(args.outpath+'artidoc_'+args.cohort+"_"+idartidoc+'_num'+'.jpg')
        scipy.misc.toimage(char).save(args.outpath+'artidoc_'+args.cohort+"_"+idartidoc+'_char'+'.jpg')
        scipy.misc.toimage(word).save(args.outpath+'artidoc_'+args.cohort+"_"+idartidoc+'_word'+'.jpg')
       
        b = bkgd*img#+ bkgd )#- num - wor  ) 
        g = bkgd*img + num*img #- bkgd - wor )
        r = bkgd*img + word*img#- num - bkgd )
        image_array = np.stack((r,g,b), axis=2)
#         scipy.misc.imsave(args.outpath+idimg+'_outfile.jpg', image_array) 
        scipy.misc.toimage(image_array, cmin=0, cmax=255).save(args.outpath+'artidoc_'+args.cohort+"_"+idartidoc+'_gts'+'.jpg')
         
        b = bkgd*img#+ bkgd )#- num - wor  ) 
        g = num*img #- bkgd - wor )
        r = word*img#- num - bkgd )
        image_array = np.stack((r,g,b), axis=2)
#         scipy.misc.imsave(args.outpath+idimg+'_outfile.jpg', image_array) 
        scipy.misc.toimage(image_array, cmin=0, cmax=255).save(args.outpath+'artidoc_'+args.cohort+"_"+idartidoc+'_gt'+'.jpg')
         
        nimg += 1 
        print(nimg) 
        
        
         
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ARTIDOC procedure. generate artificial document. the current used procedure is 'randhanddoc'. the 'randhanddoc' arguments are max_size (saize of the generated image), div (let it set to 1), denx (minimum WIDTH of a cell within the randomly created grid, in which we place the IRONOFF patches), deny ((minimum HEIGHT of a cell within the randomly created grid), dropout (density of blank patches. For each grid's cell, the probability to have a blank patches is p=(dropout)/(dropout+3) ), split ('Train', 'Valid', 'Test'), csvdb (RIMES database), csvbg (BACKGROUND database), csvst (STROKES database), snr (Signal-To-Noise Ratio) ")
    parser.add_argument("-i", "--images-database",
                        dest="csvdb",
                        type=str,
                        default="./databaseImages.csv",
                        help="csv file listing files of the IRONOFF database",)
    parser.add_argument("-l", "--strokes-database",
                        dest="csvst",
                        type=str,
                        default="./databaseStrokes.csv",
                        help="csv file listing files of the STROKES database",)
    parser.add_argument("-b", "--background-database",
                        dest="csvbg",
                        type=str,
                        default="./databaseBackground.csv",
                        help="csv file listing files of the BACKGROUND database",)
    parser.add_argument("-o", "--output-path",
                        dest="outpath",
                        type=str,
                        default="./",
                        help="output path to save results (npz numpy pickle)",)
    parser.add_argument("-r", "--noise-db",
                        dest="snr",
                        type=float, 
                        default=100,
                        help="signal to noise  to ratio",)
    parser.add_argument("-n", "--number-images",
                        dest="nbimg",
                        type=int, 
                        default=1,
                        help="",)
    parser.add_argument("-c", "--cohort",
                        dest="cohort",
                        type=str, 
                        default="Train",
                        help="",)
    parser.add_argument("-s", "--input-size",
                        dest="imgsize",
                        type=int, 
                        nargs=2,
                        default=(1024, 1024),
                        help="",)
    parser.add_argument("-g", "--generation-parameters",
                        dest="genparam",
                        type=int, 
                        nargs=3,
                        default=(200, 100, 1),
                        help="",)
    args = parser.parse_args()
    main(args)
