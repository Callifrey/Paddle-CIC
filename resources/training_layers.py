# coding:utf-8
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import numpy as np
import paddle.nn.functional as F
import sklearn.neighbors as nn
from skimage import color
'''
The code is borrowed from the original caffe implement: https://github.com/richzhang/colorization/blob/caffe/resources/caffe_traininglayers.py
'''

class NNEncLayer(object):  # nearest neighbors encode layer
    ''' Layer which encodes ab map into Q colors
    OUTPUTS
        top[0].data     NxQ
    '''
    def __init__(self, NN=5, sigma=5):
        self.NN = NN
        self.sigma = sigma
        self.nnenc = NNEncode(self.NN, self.sigma, km_filepath='./resources/pts_in_hull.npy')

        self.X = 224
        self.Y = 224
        self.Q = self.nnenc.K

    def forward(self, x):
        # return np.argmax(self.nnenc.encode_points_mtx_nd(x), axis=1).astype(np.int32)
        encode = self.nnenc.encode_points_mtx_nd(x)
        max_encode = np.argmax(encode, axis=1).astype(np.int32)
        return encode, max_encode

    def reshape(self, bottom, top):
        top[0].reshape(self.N, self.Q, self.X, self.Y)


class PriorBoostLayer(object):
    ''' Layer boosts ab values based on their rarity
    INPUTS
        bottom[0]       NxQxXxY
    OUTPUTS
        top[0].data     Nx1xXxY
    '''

    def __init__(self, ENC_DIR='./resources/', gamma=0.5, alpha=1.0):
        self.gamma = gamma
        self.alpha = alpha
        self.pc = PriorFactor(self.alpha, gamma=self.gamma, priorFile=os.path.join(ENC_DIR, 'prior_probs.npy'))

        self.X = 224
        self.Y = 224

    def forward(self, bottom):
        return self.pc.forward(bottom, axis=1)


class NonGrayMaskLayer(object):
    ''' Layer outputs a mask based on if the image is grayscale or not
    INPUTS
        bottom[0]       Nx2xXxY     ab values
    OUTPUTS
        top[0].data     Nx1xXxY     1 if image is NOT grayscale
                                    0 if image is grayscale
    '''

    def setup(self, bottom, top):
        if len(bottom) == 0:
            raise Exception("Layer should have inputs")

        self.thresh = 5  # threshold on ab value
        self.N = bottom.data.shape[0]
        self.X = bottom.data.shape[2]
        self.Y = bottom.data.shape[3]

    def forward(self, bottom):
        bottom = bottom.numpy()
        # if an image has any (a,b) value which exceeds threshold, output 1
        return (np.sum(np.sum(np.sum((np.abs(bottom) > 5).astype('float'), axis=1), axis=1), axis=1) > 0)[:,
               na(), na(), na()].astype('float')


# ***************************
# ***** SUPPORT CLASSES *****
# ***************************
class PriorFactor():
    ''' Class handles prior factor '''

    def __init__(self, alpha, gamma=0.0, verbose=True, priorFile=''):
        # INPUTS
        #   alpha           integer     prior correction factor, 0 to ignore prior, 1 to divide by prior, alpha to divide by prior**alpha
        #   gamma           integer     percentage to mix in uniform prior with empirical prior
        #   priorFile       file        file which contains prior probabilities across classes

        # settings
        self.alpha = alpha
        self.gamma = gamma
        self.verbose = verbose

        # empirical prior probability
        self.prior_probs = np.load(priorFile)

        # define uniform probability
        self.uni_probs = np.zeros_like(self.prior_probs)
        self.uni_probs[self.prior_probs != 0] = 1.
        self.uni_probs = self.uni_probs / np.sum(self.uni_probs)

        # convex combination of empirical prior and uniform distribution
        self.prior_mix = (1 - self.gamma) * self.prior_probs + self.gamma * self.uni_probs

        # set prior factor
        self.prior_factor = self.prior_mix ** -self.alpha
        self.prior_factor = self.prior_factor / np.sum(self.prior_probs * self.prior_factor)  # re-normalize

        # implied empirical prior
        self.implied_prior = self.prior_probs * self.prior_factor
        self.implied_prior = self.implied_prior / np.sum(self.implied_prior)  # re-normalize

        # if (self.verbose):
        #    self.print_correction_stats()

    def print_correction_stats(self):
        print('Prior factor correction:')
        print('  (alpha,gamma) = (%.2f, %.2f)' % (self.alpha, self.gamma))
        print('  (min,max,mean,med,exp) = (%.2f, %.2f, %.2f, %.2f, %.2f)' % (
            np.min(self.prior_factor), np.max(self.prior_factor), np.mean(self.prior_factor),
            np.median(self.prior_factor),
            np.sum(self.prior_factor * self.prior_probs)))

    def forward(self, data_ab_quant, axis=1):
        data_ab_maxind = np.argmax(data_ab_quant, axis=axis)
        corr_factor = self.prior_factor[data_ab_maxind]
        if (axis == 0):
            return corr_factor[na(), :]
        elif (axis == 1):
            return corr_factor[:, na(), :]
        elif (axis == 2):
            return corr_factor[:, :, na(), :]
        elif (axis == 3):
            return corr_factor[:, :, :, na()]


class NNEncode():
    ''' Encode points using NN search and Gaussian kernel '''

    def __init__(self, NN, sigma, km_filepath='', cc=-1):
        if (check_value(cc, -1)):
            self.cc = np.load(km_filepath)
        else:
            self.cc = cc
        self.K = self.cc.shape[0]
        self.NN = int(NN)
        self.sigma = sigma
        self.nbrs = nn.NearestNeighbors(n_neighbors=NN, algorithm='ball_tree').fit(self.cc)

        self.alreadyUsed = False

    def encode_points_mtx_nd(self, pts_nd, axis=1, returnSparse=False, sameBlock=True):
        pts_flt = flatten_nd_array(pts_nd, axis=axis)
        P = pts_flt.shape[0]
        if (sameBlock and self.alreadyUsed):
            self.pts_enc_flt[...] = 0  # already pre-allocated
        else:
            self.alreadyUsed = True
            self.pts_enc_flt = np.zeros((P, self.K))
            self.p_inds = np.arange(0, P, dtype='int')[:, na()]

        (dists, inds) = self.nbrs.kneighbors(pts_flt)

        wts = np.exp(-dists ** 2 / (2 * self.sigma ** 2))
        wts = wts / np.sum(wts, axis=1)[:, na()]
        self.pts_enc_flt[self.p_inds, inds] = wts

        pts_enc_nd = unflatten_2d_array(self.pts_enc_flt, pts_nd, axis=axis)
        return pts_enc_nd

    def decode_points_mtx_nd(self, pts_enc_nd, axis=1):
        pts_enc_flt = flatten_nd_array(pts_enc_nd, axis=axis)
        pts_dec_flt = np.dot(pts_enc_flt, self.cc)
        pts_dec_nd = unflatten_2d_array(pts_dec_flt, pts_enc_nd, axis=axis)
        return pts_dec_nd



# *****************************
# ***** Utility functions *****
# *****************************
def check_value(inds, val):
    ''' Check to see if an array is a single element equaling a particular value
    for pre-processing inputs in a function '''
    if (np.array(inds).size == 1):
        if (inds == val):
            return True
    return False


def na():  # shorthand for new axis
    return np.newaxis

def flatten_nd_array(pts_nd, axis=1):
    ''' Flatten an nd array into a 2d array with a certain axis
    INPUTS
        pts_nd       N0xN1x...xNd array
        axis         integer
    OUTPUTS
        pts_flt     prod(N \ N_axis) x N_axis array     '''
    NDIM = pts_nd.dim()
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0, NDIM), np.array((axis)))  # non axis indices
    NPTS = np.prod(SHP[nax])
    axorder = tuple(np.concatenate((nax, np.array(axis).flatten()), axis=0).tolist())
    pts_flt = pts_nd.transpose(axorder)
    pts_flt = pts_flt.reshape([NPTS.item(), SHP[axis].item()])
    return pts_flt

def unflatten_2d_array(pts_flt, pts_nd, axis=1, squeeze=False):
    ''' Unflatten a 2d array with a certain axis
    INPUTS
        pts_flt     prod(N \ N_axis) x M array
        pts_nd      N0xN1x...xNd array
        axis        integer
        squeeze     bool     if true, M=1, squeeze it out
    OUTPUTS
        pts_out     N0xN1x...xNd array        '''
    NDIM = pts_nd.dim()
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0, NDIM), np.array((axis)))  # non axis indices

    if (squeeze):
        axorder = nax
        axorder_rev = np.argsort(axorder)
        NEW_SHP = SHP[nax].tolist()

        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)
    else:
        axorder = np.concatenate((nax, np.array(axis).flatten()), axis=0)
        axorder_rev = tuple(np.argsort(axorder).tolist())
        M = pts_flt.shape[1]
        NEW_SHP = SHP[nax].tolist()
        NEW_SHP.append(M)
        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)
    return pts_out

# test阶段负责将color distribution变成一个点估计，即得到最终的ab通道
def decode(data_l, conv8_313, rebalance=2.63):
    data_l = data_l + 50
    data_l = data_l.cpu().numpy().transpose((1, 2, 0))
    enc_dir = './resources'
    conv8_313_rh = conv8_313 * rebalance
    class8_313_rh = F.softmax(conv8_313_rh, axis=0).cpu().numpy().transpose((1, 2, 0))
    cc = np.load(os.path.join(enc_dir, 'pts_in_hull.npy'))
    data_ab = np.dot(class8_313_rh, cc)
    data_ab = data_ab.repeat(4, axis=0).repeat(4, axis=1)

    img_lab = np.concatenate((data_l, data_ab), axis=-1)
    img_rgb = color.lab2rgb(img_lab)

    return img_rgb
