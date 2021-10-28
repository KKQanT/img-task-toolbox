### code from https://machinelearningmastery.com/cyclegan-tutorial-with-keras/

import tensorflow.keras.backend as K
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

import numpy as np

import matplotlib.pyplot as plt

import pickle

def load_data(filepath, domainA, domainB):
  data = np.load(filepath)
  XA, XB = data[domainA], data[domainB]
  XA = (XA-127.5)/127.5
  XB = (XB-127.5)/127.5
  return [XA, XB]

def generate_real_samples(X, n_samples, n_patch):
  idx = np.random.randint(0, X.shape[0], n_samples)
  X_real = X[idx]
  y_real = np.ones((n_samples, n_patch, n_patch, 1))
  return X_real, y_real

def generate_fake_samples(g_model, X_real, n_patch):
  X_fake = g_model.predict(X_real)
  y_fake = np.zeros((X_real.shape[0], n_patch, n_patch, 1))
  return X_fake, y_fake
  
def save_generators(step, g_model_AtoB, g_model_BtoA, mainpath):

  filenameAtoB = mainpath + f'generators/g_model_AtoB_w_{step}.h5'
  filenameBtoA = mainpath + f'generators/g_model_BtoA_w_{step}.h5'

  g_model_AtoB.save_weights(filenameAtoB)
  g_model_BtoA.save_weights(filenameBtoA)

  #symbolic_weights_AtoB = getattr(g_model_AtoB.optimizer, 'weights')
  #symbolic_weights_BtoA = getattr(g_model_BtoA.optimizer, 'weights')
#
  #weight_values_AtoB = K.batch_get_value(symbolic_weights_AtoB)
  #weight_values_BtoA = K.batch_get_value(symbolic_weights_BtoA)
#
  #with open(mainpath + f'generators/g_model_AtoB_opt.pkl', 'wb') as f:
  #  pickle.dump(weight_values_AtoB, f)
#
  #with open(mainpath + f'generators/g_model_BtoA_opt.pkl', 'wb') as f:
  #  pickle.dump(weight_values_BtoA, f)

def save_discriminators(step, d_model_A, d_model_B, mainpath):

  filenameA = mainpath + f'discriminators/d_model_A_w.h5'
  filenameB = mainpath + f'discriminators/d_model_B_w.h5'

  d_model_A.save_weights(filenameA)
  d_model_B.save_weights(filenameB)

  symbolic_weights_A = getattr(d_model_A.optimizer, 'weights')
  symbolic_weights_B = getattr(d_model_B.optimizer, 'weights')

  weight_values_A = K.batch_get_value(symbolic_weights_A)
  weight_values_B = K.batch_get_value(symbolic_weights_B)

  with open(mainpath + f'discriminators/d_model_A_opt.pkl', 'wb') as f:
    pickle.dump(weight_values_A, f)

  with open(mainpath + f'discriminators/d_model_B_opt.pkl', 'wb') as f:
    pickle.dump(weight_values_B, f)

def save_composites(step, c_model_AtoB, c_model_BtoA, mainpath):

  filenameAtoB = mainpath + f'composites/c_model_AtoB_w.h5'
  filenameBtoA = mainpath + f'composites/c_model_BtoA_w.h5'

  c_model_AtoB.save_weights(filenameAtoB)
  c_model_BtoA.save_weights(filenameBtoA)

  symbolic_weights_AtoB = getattr(c_model_AtoB.optimizer, 'weights')
  symbolic_weights_BtoA = getattr(c_model_BtoA.optimizer, 'weights')

  weight_values_AtoB = K.batch_get_value(symbolic_weights_AtoB)
  weight_values_BtoA = K.batch_get_value(symbolic_weights_BtoA)

  with open(mainpath + f'composites/c_model_AtoB_opt.pkl', 'wb') as f:
    pickle.dump(weight_values_AtoB, f)

  with open(mainpath + f'composites/c_model_BtoA_opt.pkl', 'wb') as f:
    pickle.dump(weight_values_BtoA, f)

def save_images(step, g_model, X, n_samples, mainpath, name):
  X_in, _ = generate_real_samples(X, n_samples, 0)
  X_out = g_model.predict(X_in)

  X_in = (X_in+1)/2
  X_out = (X_out+1)/2

  f,axs = plt.subplots(2, n_samples, figsize=(15,7))
  for i in range(n_samples):
    axs[0, i].imshow(X_in[i,:,:,0], cmap='gray')
    axs[1, i].imshow(X_out[i,:,:,0], cmap='gray')
  f.savefig(mainpath + f'samples/{name}_{step}.png')
  plt.close()

def update_image_pool(pool, images, max_size=50):
  selected = []
  for image in images:
    if len(pool) < max_size:
      pool.append(image)
      selected.append(image)
    elif np.random.random() < 0.5:
      selected.append(image)
    else:
      ix = np.random.randint(0, len(pool))
      selected.append(pool[ix])
      pool[ix] = image
  return np.asarray(selected)