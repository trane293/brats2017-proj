import h5py
from modules.configfile import config
import logging
from mayavi import mlab
import numpy as np

logging.basicConfig(level=logging.INFO)
try:
    logger = logging.getLogger(__file__.split('/')[-1])
except:
    logger = logging.getLogger(__name__)

# Create the data.
from numpy import pi, sin, cos, mgrid
dphi, dtheta = pi/250.0, pi/250.0
[phi,theta] = mgrid[0:pi+dphi*1.5:dphi,0:2*pi+dtheta*1.5:dtheta]
m0 = 4; m1 = 3; m2 = 2; m3 = 3; m4 = 6; m5 = 2; m6 = 6; m7 = 4;
r = sin(m0*phi)**m1 + cos(m2*phi)**m3 + sin(m4*theta)**m5 + cos(m6*theta)**m7
x = r*sin(phi)*cos(theta)
y = r*cos(phi)
z = r*sin(phi)*sin(theta)

# open the dataset

hdf5_file = h5py.File(config['hdf5_filepath_prefix'], mode='r')

# lets get a segmentation
brain = hdf5_file["training_data_segmasks_hgg"][0]

# lets get a brain
# brain = hdf5_file["training_data_hgg"][0,1,...]

xx, yy, zz = np.where(brain > 0)

mlab.points3d(xx, yy, zz,
                     mode="cube",
                     color=(0, 1, 0),
                     scale_factor=1)

mlab.show()

# View it.

s = mlab.mesh(x, y, z)
mlab.show()
hdf5_file.close()