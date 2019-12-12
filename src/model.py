import os
import time
import tensorflow as tf
import numpy as np
from collections import namedtuple
import h5py
from math import floor
from random import shuffle, choice
from scipy.interpolate import griddata
from devito import TimeFunction, clear_cache
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import Model, RickerSource, Receiver, TimeAxis
from module import *
from utils import *
import odl
from ODLscripts.odl_operators import *
from ODLscripts.layers_tf_devito import as_tensorflow_layer
import odl.contrib.tensorflow
from velGenerator import *

class LearnedWaveSim(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.image_size0 = args.image_size0
        self.image_size1 = args.image_size1
        self.input_c_dim = args.input_nc
        self.output_c_dim = args.output_nc
        self.experiment_dir = args.experiment_dir
        self.log_dir = args.log_dir
        self.sample_dir = args.sample_dir
        self.same_model_training = args.same_model_training
        self.correction_num = args.correction_num
        self.training_fraction = args.training_fraction
        self.netEpoch = args.netEpoch
        self.data_path = args.data_path

        print((" [*] Generating training velocity models..."))
        velFileGen(same_model_training=self.same_model_training, datapath=self.data_path)

        self.generator = generator_resnet
        self.criterionGAN = mae_criterion

        OPTIONS = namedtuple('OPTIONS', 'image_size0 image_size1 \
                              gf_dim df_dim output_c_dim is_training')
        self.options = OPTIONS._make((args.image_size0, args.image_size1,
                                      args.ngf, args.ndf, args.output_nc,
                                      args.phase == 'train'))

        self._setup_devito(velIndex=0)
        self._build_model()
        self.saver = tf.train.Saver()

    def _build_model(self):

        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.SNR_diff = tf.placeholder(tf.float32, [None, self.image_size0 * \
            self.image_size1 * self.output_c_dim], name='SNR_diff')
        self.SNR_real = tf.placeholder(tf.float32, [None, self.image_size0 * \
            self.image_size1 * self.output_c_dim], name='SNR_real')

        self.LF_wave  = tf.placeholder(tf.float32, shape=[None, self.image_size0, \
            self.image_size1, self.input_c_dim], name='LF_wave')
        self.HF_wave  = tf.placeholder(tf.float32, shape=[None, self.image_size0, \
            self.image_size1, self.input_c_dim], name='HF_wave')

        self.Rec_SNR = -20.0* tf.log(tf.norm(self.SNR_diff, ord='euclidean')/tf.norm(self.SNR_real, 
            ord='euclidean'))/tf.log(10.0)

        self.fwd_op = as_tensorflow_layer(F(self.model, self.src_zero, \
            self.rec_zero.coordinates.data, time_start=0, 
            time_end=self.virt_timestep, space_order=2), 'Forward')

        self.g_loss   = []
        self.d_loss   = []
        self.g_sum    = []
        self.d_sum    = []
        self.SNR_sum  = []
        self.CNN_wave = []
        self.Noisy_wave = []
        
        iG = 0
        self.Noisy_wave.append(self.LF_wave)
        self.CNN_wave.append(self.generator(self.Noisy_wave[iG], self.options, False, \
            name="generator_" + str(iG)))

        self.g_loss.append(abs_criterion(self.CNN_wave[iG], self.HF_wave))
        
        self.g_sum.append(tf.summary.scalar("g_loss_" + str(iG), self.g_loss[iG]))
        self.SNR_sum.append(tf.summary.scalar("SNR_" + str(iG), self.Rec_SNR))

        for iG in range(1, self.correction_num):

            self.Noisy_wave.append(tf.concat(tf.split(self.fwd_op(tf.concat(tf.split(\
                self.CNN_wave[iG-1], 3, axis=3), axis=1)), 3, axis=1), axis=3))
            self.CNN_wave.append(self.generator(self.Noisy_wave[iG], self.options, False, \
                name="generator_" + str(iG)))
            
            self.g_loss.append(abs_criterion(self.CNN_wave[iG], self.HF_wave))

            self.g_sum.append(tf.summary.scalar("g_loss_" + str(iG), self.g_loss[iG]))
            self.SNR_sum.append(tf.summary.scalar("SNR_" + str(iG), self.Rec_SNR))

        self.t_vars = tf.trainable_variables()
        self.g_vars = []
        for iG in range(self.correction_num):
            self.g_vars.append([var for var in self.t_vars if 'generator_' + str(iG) in var.name])

        var_size = 0
        for var in self.t_vars:
            var_size = var_size + int(np.prod(np.array(var.shape)))
        print(("Number of unknowns: %d" % (var_size)))

    def _setup_devito(self, velIndex=0):

        origin = (0.0, 0.0)
        self.spacing = (7.5, 7.5)
        self.nbpml = 40
        self.num_rec = 401
        self.shape = [self.image_size0-2*self.nbpml , self.image_size1-2*self.nbpml]
        tn, vp = velLoader(velIndex=velIndex, datapath=self.data_path)
        self.model = Model(origin, self.spacing, self.shape, 2, vp, nbpml=self.nbpml)
        dt = self.model.critical_dt
        t0 = 0.0
        nt = int(1 + (tn - t0) / dt)
        self.virt_timestep = int(nt // self.correction_num)
        rec_samp = np.linspace(0., self.model.domain_size[0], num=self.num_rec)
        rec_samp = rec_samp[1] - rec_samp[0]
        xsrc = 100
        time_range = TimeAxis(start=t0, stop=tn, step=dt)

        self.src = RickerSource(name='src', grid=self.model.grid, f0=0.025,
                                time_range=time_range, space_order=1, npoint=1)
        self.src.coordinates.data[0, :] = np.array([xsrc * self.spacing[0], 2 * \
            self.spacing[1]]).astype(np.float32)
        self.src_zero = RickerSource(name='src_zero', grid=self.model.grid, f0=0.025,
                                time_range=time_range, space_order=1, npoint=1)
        self.src_zero.data.fill(0.)
        self.src_zero.coordinates.data[0, :] = np.array([xsrc * self.spacing[0], 2 * \
            self.spacing[1]]).astype(np.float32)
        self.rec = Receiver(name='rec', grid=self.model.grid, time_range=time_range, \
            npoint=self.num_rec)
        self.rec.coordinates.data[:, 0] = np.linspace(0., self.model.domain_size[0], \
            num=self.num_rec)
        self.rec.coordinates.data[:, 1:] = self.src.coordinates.data[0, 1:]
        self.rec_zero = Receiver(name='rec_zero', grid=self.model.grid, time_range=time_range, \
            npoint=self.num_rec)
        self.rec_zero.coordinates.data[:, 0] = np.linspace(0., self.model.domain_size[0], \
            num=self.num_rec)
        self.rec_zero.coordinates.data[:, 1:] = self.src_zero.coordinates.data[0, 1:]

        self.solverLF = AcousticWaveSolver(self.model, source=self.src, receiver=self.rec, \
            kernel='OT2', space_order=2)
        self.solverHF = AcousticWaveSolver(self.model, source=self.src, receiver=self.rec, \
            kernel='OT2', space_order=20)

        self.u_HF = TimeFunction(name="u", grid=self.model.grid, time_order=2, space_order=20)
        self.u_LF = TimeFunction(name="u", grid=self.model.grid, time_order=2, space_order=2)

    def update_devito(self, velIndex=1):

        tn, vp = velLoader(velIndex=velIndex, datapath=self.data_path)
        self.model.vp = vp
        dt = self.model.critical_dt
        t0 = 0.0
        nt = int(1 + (tn - t0) / dt)
        self.virt_timestep = int(nt // self.correction_num)
        rec_samp = np.linspace(0., self.model.domain_size[0], num=self.num_rec)
        rec_samp = rec_samp[1] - rec_samp[0]
        xsrc = 100
        time_range = TimeAxis(start=t0, stop=tn, step=dt)

        self.src.coordinates.data[0, :] = np.array([xsrc * self.spacing[0], 2 * \
            self.spacing[1]]).astype(np.float32)
        self.src_zero.data.fill(0.)
        self.src_zero.coordinates.data[0, :] = np.array([xsrc * self.spacing[0], 2 * \
            self.spacing[1]]).astype(np.float32)

        self.rec.coordinates.data[:, 0] = np.linspace(0., self.model.domain_size[0], \
            num=self.num_rec)
        self.rec.coordinates.data[:, 1:] = self.src.coordinates.data[0, 1:]
        self.rec_zero.coordinates.data[:, 0] = np.linspace(0., self.model.domain_size[0], \
            num=self.num_rec)
        self.rec_zero.coordinates.data[:, 1:] = self.src_zero.coordinates.data[0, 1:]
        self.u_HF.data.fill(0.)
        self.u_LF.data.fill(0.)

    def train(self, args):

        self.g_optim = []
        for iG in range(self.correction_num):
            self.g_optim.append(tf.train.AdamOptimizer(self.lr, beta1=args.beta1).minimize(\
                self.g_loss[iG], 
                var_list=self.g_vars[iG]))

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        counter = 1
        start_time = time.time()

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        if self.same_model_training==0:
            velIndex = list(range(1, 6, 1))
        elif self.same_model_training==1:
            velIndex = list(range(0, 1, 1))
        batch_idxs = list(range(0, self.shape[0], self.training_fraction))

        if os.path.isfile('TempTrainingData.hdf5'):
            os.remove('TempTrainingData.hdf5')
        datasetSize = (len(velIndex)*len(batch_idxs), 1, self.image_size0, \
            self.image_size1, self.input_c_dim)        
        tempTrainingData = h5py.File('TempTrainingData.hdf5', 'w-')
        Noisy_wave = tempTrainingData.create_dataset("LWS", datasetSize)
        HF_wave = tempTrainingData.create_dataset("HF", datasetSize)

        epoch = 0
        while epoch < args.epoch:

            lr = args.lr if epoch < args.epoch_step else args.lr*(args.epoch-epoch)/(args.epoch-args.epoch_step)
            network_num = int(np.mod(floor(epoch/self.netEpoch), self.correction_num))
            shuffle(velIndex)

            tempTrainingData_itr = 0
            for velIdx in range(len(velIndex)):
                print((" [*] Updating Devito for velocity model #%1d"  % (velIndex[velIdx])))
                self.update_devito(velIndex=velIndex[velIdx])
                print((" [*] Generating training pairs for network #%1d"  % (network_num)))
                shuffle(batch_idxs)

                for isrc in range(0, len(batch_idxs)):
                    print(("Wave propagarion for veclocity #%d source #%3d and network #%2d" \
                        % (velIdx, isrc, network_num)))
                    xsrc = batch_idxs[isrc]
                    Noisy_wave[tempTrainingData_itr], HF_wave[tempTrainingData_itr] = \
                        self.wave_propagate(xsrc, network_num)
                    tempTrainingData_itr = tempTrainingData_itr + 1

            print((" [*] Training network #%1d"  % (network_num)))
            for miniepoch in range(self.netEpoch):
                for idx in range(0, len(velIndex)*len(batch_idxs)):
                    _, summary_str = self.sess.run(
                        [self.g_optim[network_num], self.g_sum[network_num]], \
                        feed_dict={self.Noisy_wave[network_num]: Noisy_wave[idx], self.HF_wave: HF_wave[idx], 
                        self.lr: lr})
                    self.writer.add_summary(summary_str, counter)
                    counter += 1
                    print(("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4f" % (
                        epoch, args.epoch, int(idx), int(len(velIndex)*len(batch_idxs)), time.time() - start_time)))
                    if np.mod(counter, args.print_freq) == 1:
                        self.sample_model(args.sample_dir, epoch, network_num, counter-1)
                    if np.mod(counter, args.save_freq) == 2:
                        self.save(args.checkpoint_dir, counter)
                epoch = epoch + 1

    def wave_propagate(self, xsrc, iG):
        clear_cache()
        self.src.coordinates.data[0, :] = np.array([xsrc*self.spacing[0], \
            2*self.spacing[1]]).astype(np.float32)
        self.u_HF.data.fill(0.)
        self.u_LF.data.fill(0.)
        self.solverHF.forward(m=self.model.m, src=self.src, time_m=0, 
            time=(iG+1)*self.virt_timestep, u=self.u_HF)
        HF_wave = np.transpose(np.array(self.u_HF.data[:, :, :]), \
            (1, 2, 0)).astype(np.float32)[None, :, :, :]
        self.solverLF.forward(m=self.model.m, src=self.src, time_m=0, 
            time=self.virt_timestep,  u=self.u_LF)
        LF_wave = np.transpose(np.array(self.u_LF.data[:, :, :]), \
            (1, 2, 0)).astype(np.float32)[None, :, :, :]
        Noisy_wave = self.sess.run(
            [self.Noisy_wave[iG]],
            feed_dict={self.LF_wave: LF_wave})[0]
        return Noisy_wave, HF_wave

    def save(self, checkpoint_dir, step):
        model_name = "LearnedWaveSim.model"
        model_dir = "%s_%s" % (self.experiment_dir, self.image_size0)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")
        model_dir = "%s_%s" % (self.experiment_dir, self.image_size0)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def sample_model(self, sample_dir, epoch, iG, counter, mask=None):

        self.update_devito(velIndex=0)
        if self.same_model_training==0:
            batch_idxs = list(range(0, self.shape[0]))
        elif self.same_model_training==1:
            batch_idxs = list(range(0, self.shape[0], self.training_fraction))

        xsrc = choice(batch_idxs)
        HF_wave_history  = []
        CNN_wave_history = []
        LF_wave_history  = []
        Rec_SNR  = []

        Noisy_wave, HF_wave = self.wave_propagate(xsrc, iG)
        CNN_wave = self.sess.run(
            [self.CNN_wave[iG]],
            feed_dict={self.Noisy_wave[iG]: Noisy_wave, self.HF_wave: HF_wave})
        diff_img = np.absolute(np.transpose(HF_wave[0, :, :, :], (2, 0 , 1)) - \
            np.transpose(CNN_wave[0][0, :, :, :], (2, 0 , 1)))
        diff_img = diff_img.reshape((1, self.image_size0 * self.image_size1 * \
            self.output_c_dim))
        snr, summary_str = self.sess.run(
        [self.Rec_SNR, self.SNR_sum[iG]],
        feed_dict={self.SNR_diff: diff_img,
                   self.SNR_real: np.transpose(HF_wave[0, :, :, :], \
                    (2, 0 , 1)).reshape((1, self.image_size0 * self.image_size1 * \
                        self.output_c_dim))})
        self.writer.add_summary(summary_str, counter)
        print(("Mapping SNR (Training data) for network #%d: %4.4f" % (iG, snr)))

    def test(self, args):

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        self.update_devito(velIndex=0)
        batch_idxs = list(range(0, self.shape[0]))
        xsrc = 201 #choice(batch_idxs)

        if not os.path.isfile(os.path.join(self.sample_dir, 'LearnedFwdSimPrediction.hdf5')):
            datasetSize = (self.output_c_dim, self.image_size0, self.image_size1, \
                self.correction_num)
            self.file_prediction = h5py.File(os.path.join(self.sample_dir, \
                'LearnedFwdSimPrediction.hdf5'), 'w-')
            self.dataset_CNN = self.file_prediction.create_dataset("result", datasetSize)
            self.dataset_HF = self.file_prediction.create_dataset("HF", datasetSize)
            self.dataset_LF = self.file_prediction.create_dataset("LF", datasetSize)
        else:
            self.file_prediction = h5py.File(os.path.join(self.sample_dir, \
                'LearnedFwdSimPrediction.hdf5'), 'r+')
            self.dataset_CNN = self.file_prediction["result"]
            self.dataset_HF = self.file_prediction["HF"]
            self.dataset_LF = self.file_prediction["LF"]

        print('Processing shot number: ' + str(xsrc))
        HF_wave_history  = []
        CNN_wave_history = []
        CNN_input_history = []
        LF_wave_history  = []
        Rec_SNR  = []

        self.src.coordinates.data[0, :] = np.array([xsrc*self.spacing[0], \
            2*self.spacing[1]]).astype(np.float32)
        self.rec.coordinates.data[:, 0] = np.linspace(0., self.model.domain_size[0], \
            num=self.num_rec)
        self.rec.coordinates.data[:, 1:] = self.src.coordinates.data[0, 1:]

        self.u_HF.data.fill(0.)
        self.u_LF.data.fill(0.)

        self.solverLF.forward(m=self.model.m, src=self.src, time_m=0, 
            time=self.virt_timestep,  u=self.u_LF)
        LF_wave = np.transpose(np.array(self.u_LF.data[:, :, :]), \
            (1, 2, 0)).astype(np.float32)[None, :, :, :]
        CNN_wave_history, CNN_input_history = self.sess.run(
            [self.CNN_wave, self.Noisy_wave],
            feed_dict={self.LF_wave: LF_wave})
        self.u_LF.data.fill(0.)
        self.u_HF.data.fill(0.)

        for time_index in range(self.correction_num):
            clear_cache()
            self.solverLF.forward(m=self.model.m, src=self.src, time_m=time_index*\
                self.virt_timestep, time=(time_index+1)*self.virt_timestep, u=self.u_LF)
            LF_wave_history.append(np.transpose(np.array(self.u_LF.data[:, :, :]), 
                (1, 2, 0)).astype(np.float32)[None, :, :, :])
            self.solverHF.forward(m=self.model.m, src=self.src, time_m=time_index*\
                self.virt_timestep, time=(time_index+1)*self.virt_timestep, u=self.u_HF)
            HF_wave_history.append(np.transpose(np.array(self.u_HF.data[:, :, :]), 
                (1, 2, 0)).astype(np.float32)[None, :, :, :])

        for iG in range(self.correction_num):
            self.dataset_CNN[:, :, :, iG] = np.transpose(CNN_wave_history[iG][0, :, :, :], \
                (2, 0 , 1))
            self.dataset_HF[:, :, :, iG] = np.transpose(HF_wave_history[iG][0, :, :, :], (2, 0 , 1))
            self.dataset_LF[:, :, :, iG] =np.transpose(CNN_input_history[iG][0, :, :, :], (2, 0 , 1))
        self.file_prediction.close()
