import numpy as np
import h5py
import os
from math import floor
from scipy.interpolate import griddata


def velGenerator(velIndex=0, datapath=os.path.join(os.path.expanduser("~"), 'data')):

    if velIndex == 0:

        vp = np.fromfile(os.path.join(datapath, 'vp_marmousi_bi'),
                    dtype='float32', sep="")
        vp = np.reshape(vp, (1601, 401))
        shape=[401, 301]
        values = np.zeros([vp.shape[0]*vp.shape[1], ])
        points = np.zeros([vp.shape[0]*vp.shape[1], 2])
        k = 0
        for indx in range(0, vp.shape[0]):
            for indy in range(0, vp.shape[1]):
                values[k] = vp[indx, indy]
                points[k, 0] = indx
                points[k, 1] = indy
                k = k + 1

        X, Y = np.meshgrid(np.array(np.linspace(500, 1300, shape[0])), np.array(np.linspace(0, 312, shape[1])))
        int_vp = griddata(points, values, (X, Y), method='cubic')
        int_vp = np.transpose(int_vp)
        vp = int_vp
        return vp


    elif velIndex == 1:

        vp = np.fromfile(os.path.join(datapath, 'vp_marmousi_bi'),
                    dtype='float32', sep="")
        vp = np.reshape(vp, (1601, 401))
        shape=[401, 301]
        values = np.zeros([vp.shape[0]*vp.shape[1], ])
        points = np.zeros([vp.shape[0]*vp.shape[1], 2])
        k = 0
        for indx in range(0, vp.shape[0]):
            for indy in range(0, vp.shape[1]):
                values[k] = vp[indx, indy]
                points[k, 0] = indx
                points[k, 1] = indy
                k = k + 1

        X, Y = np.meshgrid(np.array(np.linspace(1013, 1300, shape[0])), np.array(np.linspace(38, 150, shape[1])))
        int_vp = griddata(points, values, (X, Y), method='cubic')
        int_vp = np.transpose(int_vp)
        vp = int_vp
        return vp

    elif velIndex == 2:

        vp = np.fromfile(os.path.join(datapath, 'vp_marmousi_bi'),
                    dtype='float32', sep="")
        vp = np.reshape(vp, (1601, 401))
        shape=[401, 301]
        values = np.zeros([vp.shape[0]*vp.shape[1], ])
        points = np.zeros([vp.shape[0]*vp.shape[1], 2])
        k = 0
        for indx in range(0, vp.shape[0]):
            for indy in range(0, vp.shape[1]):
                values[k] = vp[indx, indy]
                points[k, 0] = indx
                points[k, 1] = indy
                k = k + 1

        X, Y = np.meshgrid(np.array(np.linspace(1000, 1287, shape[0])), np.array(np.linspace(120, 232, shape[1])))
        int_vp = griddata(points, values, (X, Y), method='cubic')
        int_vp = np.transpose(int_vp)
        vp = int_vp
        return vp

    elif velIndex == 3:

        vp = np.fromfile(os.path.join(datapath, 'vp_marmousi_bi'),
                    dtype='float32', sep="")
        vp = np.reshape(vp, (1601, 401))
        shape=[401, 301]
        values = np.zeros([vp.shape[0]*vp.shape[1], ])
        points = np.zeros([vp.shape[0]*vp.shape[1], 2])
        k = 0
        for indx in range(0, vp.shape[0]):
            for indy in range(0, vp.shape[1]):
                values[k] = vp[indx, indy]
                points[k, 0] = indx
                points[k, 1] = indy
                k = k + 1

        X, Y = np.meshgrid(np.array(np.linspace(730, 1017, shape[0])), np.array(np.linspace(60, 172, shape[1])))
        int_vp = griddata(points, values, (X, Y), method='cubic')
        int_vp = np.transpose(int_vp)
        vp = int_vp
        return vp

    elif velIndex == 4:

        vp = np.fromfile(os.path.join(datapath, 'vp_marmousi_bi'),
                    dtype='float32', sep="")
        vp = np.reshape(vp, (1601, 401))
        shape=[401, 301]
        values = np.zeros([vp.shape[0]*vp.shape[1], ])
        points = np.zeros([vp.shape[0]*vp.shape[1], 2])
        k = 0
        for indx in range(0, vp.shape[0]):
            for indy in range(0, vp.shape[1]):
                values[k] = vp[indx, indy]
                points[k, 0] = indx
                points[k, 1] = indy
                k = k + 1

        X, Y = np.meshgrid(np.array(np.linspace(500, 787, shape[0])), np.array(np.linspace(130, 242, shape[1])))
        int_vp = griddata(points, values, (X, Y), method='cubic')
        int_vp = np.transpose(int_vp)
        vp = int_vp
        return vp

    elif velIndex == 5:

        vp = np.fromfile(os.path.join(datapath, 'vp_marmousi_bi'),
                    dtype='float32', sep="")
        vp = np.reshape(vp, (1601, 401))
        shape=[401, 301]
        values = np.zeros([vp.shape[0]*vp.shape[1], ])
        points = np.zeros([vp.shape[0]*vp.shape[1], 2])
        k = 0
        for indx in range(0, vp.shape[0]):
            for indy in range(0, vp.shape[1]):
                values[k] = vp[indx, indy]
                points[k, 0] = indx
                points[k, 1] = indy
                k = k + 1

        X, Y = np.meshgrid(np.array(np.linspace(0, 287, shape[0])), np.array(np.linspace(200, 312, shape[1])))
        int_vp = griddata(points, values, (X, Y), method='cubic')
        int_vp = np.transpose(int_vp)
        vp = int_vp
        return vp


def velFileGen(same_model_training=1, datapath=os.path.join(os.path.expanduser("~"), 'data')):

    if not os.path.isfile(os.path.join(datapath, 'marmousi-trainingCrops.hdf5')):
        strName = os.path.join(datapath, 'marmousi-trainingCrops.hdf5')
        dataset_name = "model_"
        fileName = h5py.File(strName, 'w-')
        
        if not same_model_training:
            for i in range(6):
                print(("        Velocity model #[%1d]"  % (i)))
                shape=[401, 301]
                dataGradientsA = fileName.create_dataset(dataset_name + str(i), (shape[0], shape[1]))
                vp = velGenerator(velIndex=i, datapath=datapath)
                dataGradientsA[:,:] = vp
        else:
            i = 0
            print(("        Velocity model #[%1d]"  % (i)))
            shape=[401, 301]
            dataGradientsA = fileName.create_dataset(dataset_name + str(i), (shape[0], shape[1]))
            vp = velGenerator(velIndex=i, datapath=datapath)
            dataGradientsA[:,:] = vp            
    else:
        print((" [*] Training velocity models already exist."))


def velLoader(velIndex=0, datapath=os.path.join(os.path.expanduser("~"), 'data')):
    strName = os.path.join(datapath, 'marmousi-trainingCrops.hdf5')
    dataset_name = "model_"
    file_name = h5py.File(strName, 'r')

    if velIndex == 0:
        origin = (0., 0.)
        spacing=(7.5, 7.5)
        tn=1100.
        nbpml=40
        vp  = file_name[dataset_name + str(velIndex)][:,:]
        file_name.close()
        return tn, vp

    elif velIndex == 1:
        origin = (0., 0.)
        spacing=(7.5, 7.5)
        tn=1100.
        nbpml=40
        vp  = file_name[dataset_name + str(velIndex)][:,:]
        file_name.close()
        return tn, vp

    elif velIndex == 2:
        origin = (0., 0.)
        spacing=(7.5, 7.5)
        tn=1100.
        nbpml=40
        vp  = file_name[dataset_name + str(velIndex)][:,:]
        file_name.close()
        return tn, vp

    elif velIndex == 3:
        origin = (0., 0.)
        spacing=(7.5, 7.5)
        tn=1100.
        nbpml=40
        vp  = file_name[dataset_name + str(velIndex)][:,:]
        file_name.close()
        return tn, vp

    elif velIndex == 4:
        origin = (0., 0.)
        spacing=(7.5, 7.5)
        tn=1100.
        nbpml=40
        vp  = file_name[dataset_name + str(velIndex)][:,:]
        file_name.close()
        return tn, vp

    elif velIndex == 5:
        origin = (0., 0.)
        spacing=(7.5, 7.5)
        tn=1100.
        nbpml=40
        vp  = file_name[dataset_name + str(velIndex)][:,:]
        file_name.close()
        return tn, vp