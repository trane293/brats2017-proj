# Modular Framework for BRATS 2017 Tumor Segmentation Experiments

This framework is being developed as part of my masters research in segmenting tumors in Brain MRI scans.
We use the BRATS 2017 dataset for our work, and the framework allows working off of this dataset in a very
memory-friendly way. The core of the framework is based upon the HDF5 data storage, which allows asynchronous
I/O to the disk with a highly abstracted interface which makes it appear as the data was in memory. This also
brings other benfits with it such as hosting the data on a cloud server, and accessing it over SSHFS.

Another aspect of this framework is to allow easy plug-and-play experimentation. Many machine learning/deep learning
researchers that I meet usually write their code in Jupyter Notebook, which although provides a great deal of flexibility,
also leads to massive speghetti code once the codebase becomes bigger. Also, the need to manage experiments effectively,
correctly, and in a reproducible way is the need of the hour. The framework is desgined with this ideology in mind,
where each experiment involving a certain iteration of a network will be self-contained. The folder `defmodel` holds
individual network configurations, along with mandatory helper functions defined in a self-contained way. This allows
easy reproducibility of experiments since hyperparameters are decoupled from the training script, as well as overall
cleaner and neater code.

# Folder Structure

```
code
- defmodel
 - Contains standalone model/network definitions along with helper functions. Check cnn_patches.py for example.
- graph
 - Contains log folders for each experiment, useful for visualizing training through tensorboard.
-legacy
 - Old code which may/may not be useful today.  Good for reference.
- modules
 - Contains modules which define many helper functions ranging from loading data, preprocessing, training, visualization and validation.
- notebooks
 - Quick prototyping notebooks. Very rough around the edges, good for reference, not used in framework.
- test_files
 - Contains files to test the data loader functions through manual testing and visualization.
- viz_images
 - Contains images generated during validation phase.

create_hdf5_file.py
- File used to generate the HDF5 data store for the BRATS 2017 dataset. Please check config.py in `modules` before proceeding.
preprocess.py
- Preprocess data using input HDF5 data store, and write a new HDF5 store with preprocessed images. The script i) crops out brain region, and ii) calculates mean/var of the patient data.
train_seg.py
- Main file used to train any arbitrary model/network defined inside the defmodel folder. Use --help to know more about command line parameters
validate.py
- Validate pre-trained models on BRATS 2017 Validation Data.
```

