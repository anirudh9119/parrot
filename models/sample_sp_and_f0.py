import numpy
import matplotlib
matplotlib.use('Agg')
import sys

import scipy#, pylab
from audio_tools import soundsc
#from statistics import variance, stdev
from scipy.io import wavfile

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append('../../../../')
sys.path.append('../../../../../')

#from matplotlib import pyplot
import os
#from play.utils.mgc import mgcf02wav
#import pysptk as SPTK
#from scipy.io import wavfile
from blocks.serialization import load
from blocks.graph import ComputationGraph
#import ipdb

from fuel.transformers import (Mapping, ForceFloatX)  #FilterSources ScaleAndShift
from fuel.schemes import SequentialScheme
from fuel.streams import DataStream
#from play.datasets.blizzard import Blizzard
from fuel.datasets import H5PYDataset
from play.toy.segment_transformer import SegmentSequence

batch_size = 10

order = 34
alpha = 0.4
stage = 2
gamma = -1.0 / stage


def istft(X, fs, T, hop):
    x = scipy.zeros(T*fs)
    framesamp = X.shape[1]
    hopsamp = int(hop*fs)
    for n,i in enumerate(range(0, len(x)-framesamp, hopsamp)):
        x[i:i+framesamp] += scipy.real(scipy.ifft(X[n]))
    return x


def _transpose(data):
    return tuple(array.swapaxes(0,1) for array in data)

def _is_nonzero(data):
    return tuple([1.*(data[0]>0)])

def _zero_for_unvoiced(data):
    #Multiply by 0 the unvoiced components.
    return tuple([data[0]*data[2],data[1],data[2]])

#data_dir = os.environ['FUEL_DATA_PATH']

data_dir = '/data/lisatmp4/sotelo/data/'
data_dir = os.path.join(data_dir, 'blizzard/', 'sp_standardize.npz')

data_stats = numpy.load(data_dir)
sp_mean = data_stats['sp_mean']
sp_std = data_stats['sp_std']
f0_mean = data_stats['f0_mean']
f0_std = data_stats['f0_std']

#dataset = Blizzard(which_sets = ('test',), filename = "sp_blizzard.hdf5")

dataset = H5PYDataset("/data/lisatmp4/sotelo/data/blizzard/chunk_0.hdf5", ('test',), )

data_stream = DataStream.default_stream(
            dataset, iteration_scheme=SequentialScheme(
            batch_size*(dataset.num_examples/batch_size), batch_size))

'''
data_stream = Mapping(data_stream, _is_nonzero, add_sources = ('voiced',))


data_stream = ScaleAndShift(data_stream,
                            scale = 1/sp_std,
                            shift = -sp_mean/sp_std,
                            which_sources = ('sp',))
data_stream = ScaleAndShift(data_stream,
                            scale = 1/f0_std,
                            shift = -f0_mean/f0_std,
                            which_sources = ('f0',))
data_stream = Mapping(data_stream, _zero_for_unvoiced)

'''

data_stream = Mapping(data_stream, _transpose)
data_stream = SegmentSequence(data_stream, 128*2, add_flag=True)
data_stream = ForceFloatX(data_stream)

x_tr = next(data_stream.get_epoch_iterator())


#save_dir = os.environ['RESULTS_DIR']

save_dir = '/Tmp/anirudhg/results/'
save_dir = os.path.join(save_dir,'blizzard/')

experiment_name = "best_baseline_sp_"
num_sample = "1"

main_loop = load(save_dir + "pkl/" + experiment_name + ".pkl")

generator = main_loop.model.get_top_bricks()[0]

steps = 183
n_samples = 10

from blocks.utils import shared_floatx_zeros #shared_floatx

hidden_size_recurrent = 2000

states = generator.transition.apply.outputs
states = {name: shared_floatx_zeros((batch_size, hidden_size_recurrent)) for name in states}

from theano import tensor #function

'''
f0 = tensor.matrix('f0')
start_flag = tensor.scalar('start_flag')
voiced = tensor.matrix('voiced')
sp = tensor.tensor3('sp')

f0s = f0.dimshuffle(0,1,'x')
voiceds = voiced.dimshuffle(0,1,'x')
x = tensor.concatenate([sp, f0s, voiceds], 2)

'''
start_flag = tensor.scalar('start_flag')
f0 = tensor.tensor3('amplitude')
sp = tensor.tensor3('phase')
sps = sp.dimshuffle(1,0,2)
f0s = f0.dimshuffle(1,0,2)
x = tensor.concatenate([f0s, sps], 2)



cost_matrix = generator.cost_matrix(x)

cg = ComputationGraph(cost_matrix)


#generator.generate(n_steps=steps, batch_size=n_samples, iterate=True, **states)

#states = {}
sample = ComputationGraph(generator.generate(n_steps=steps,
    batch_size=n_samples, iterate=True, **states))
sample_fn = sample.get_theano_function()

outputs_bp = sample_fn()[-2]

for this_sample in range(n_samples):
	print "Iteration: ", this_sample
	outputs = outputs_bp
        outputs2 = outputs
        print outputs.shape


        outputs = outputs[:,:, :-401]

	outputs = outputs.swapaxes(0,1)
	outputs = outputs[this_sample]
        print "Amlitude Values: ", outputs.max(), outputs.min()
        amplitude = outputs
        amplitude = numpy.hstack([amplitude, amplitude[:,::-1][:,1:-1]])
        amplitude = amplitude.astype('float64').copy(order = 'C')
        print "Amplitude Shape: ", amplitude.shape

        outputs_phase = outputs2[:,:, 401:]
	outputs_phase = outputs_phase.swapaxes(0,1)
	outputs_phase = outputs_phase[this_sample]
        print "Phase values: ", outputs_phase.max(), outputs_phase.min()
        phase = outputs_phase
        phase = numpy.hstack([phase, phase[:,::-1][:,1:-1]])
        phase = phase.astype('float64').copy(order = 'C')
        print "Phase Shape", phase.shape

#        numpy.savez('temp.npz',amplitude, phase)

        ### amplitude/phase - 2048 * 800

        x_temp = []

        for i in range(len(amplitude)):
            temp = amplitude[i][:]
            b=[]
            for j in range(len(temp)):
                b.append(complex(temp[j], phase[i][j]))
            x_temp.append(b)


        #What should be T ?
        T = 4.61 #Should be length of sequence for ex 60 sec or 2 minutes..
        x_temp = numpy.asarray(x_temp)
        x_b = istft(x_temp, 16000, T,  0.025)
        wavfile.write("some_name_" + str(this_sample) + ".wav", 16000, soundsc(x_b))


        '''
	sampled_f0 = sampled_f0*f0_std + f0_mean
	sampled_f0 = sampled_f0*sampled_voiced
	sampled_f0 = sampled_f0.swapaxes(0,1)
	sampled_f0 = sampled_f0[this_sample]

	print sampled_f0.min(), sampled_f0.max()

	f, axarr = pyplot.subplots(2, sharex=True)
	f.set_size_inches(100,35)
	axarr[0].imshow(outputs.T)
	#axarr[0].colorbar()
	axarr[0].invert_yaxis()
	axarr[0].set_ylim(0,257)
	axarr[0].set_xlim(0,2048)
	axarr[1].plot(sampled_f0,linewidth=3)
	axarr[0].set_adjustable('box-forced')
	axarr[1].set_adjustable('box-forced')
	pyplot.savefig(save_dir+"samples/best_"+experiment_name+num_sample+str(this_sample)+".png")
	pyplot.close()

	sampled_f0_corrected = sampled_f0
	sampled_f0_corrected[sampled_f0_corrected<0] = 0.

	mgc_sp = outputs
	mgc_sp_test = numpy.hstack([mgc_sp,mgc_sp[:,::-1][:,1:-1]])
	mgc_sp_test = mgc_sp_test.astype('float64').copy(order = 'C')

	mgc_reconstruct = numpy.apply_along_axis(SPTK.mgcep, 1, mgc_sp_test, order, alpha, gamma, eps = 0.0012, etype = 1, itype = 2)

	x_synth = mgcf02wav(mgc_reconstruct, sampled_f0_corrected)
	x_synth = .95 * x_synth/max(abs(x_synth)) * 2**15
	wavfile.write(save_dir+"samples/best_"+experiment_name+num_sample+str(this_sample)+ ".wav", 16000, x_synth.astype('int16'))

	#Scaling
	outputs[outputs>11.866405] = 11.866405
	outputs[outputs<-2.0992377] = -2.0992377

	f, axarr = pyplot.subplots(2, sharex=True)
	f.set_size_inches(100,35)
	axarr[0].imshow(outputs.T)
	#axarr[0].colorbar()
	axarr[0].invert_yaxis()
	axarr[0].set_ylim(0,257)
	axarr[0].set_xlim(0,2048)
	axarr[1].plot(sampled_f0,linewidth=3)
	axarr[0].set_adjustable('box-forced')
	axarr[1].set_adjustable('box-forced')
	pyplot.savefig(save_dir+"samples/best_"+experiment_name+num_sample+str(this_sample)+"_scaled.png")
	pyplot.close()

	mgc_sp = outputs
	mgc_sp_test = numpy.hstack([mgc_sp,mgc_sp[:,::-1][:,1:-1]])
	mgc_sp_test = mgc_sp_test.astype('float64').copy(order = 'C')
	mgc_reconstruct = numpy.apply_along_axis(SPTK.mgcep, 1, mgc_sp_test, order, alpha, gamma, eps = 0.0012, etype = 1, itype = 2)
	x_synth = mgcf02wav(mgc_reconstruct, sampled_f0_corrected)
	x_synth = .95 * x_synth/max(abs(x_synth)) * 2**15
	wavfile.write(save_dir+"samples/best_"+experiment_name+num_sample+str(this_sample)+ "_scaled.wav", 16000, x_synth.astype('int16'))

        '''
