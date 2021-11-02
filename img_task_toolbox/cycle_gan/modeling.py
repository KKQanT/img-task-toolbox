### code from https://machinelearningmastery.com/cyclegan-tutorial-with-keras/

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

def build_discriminator(input_shape):

  kernel_initializer = RandomNormal(stddev=0.02)

  input_ = L.Input(input_shape)

  x = L.Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=kernel_initializer)(input_)
  x = L.LeakyReLU(0.2)(x)

  x = L.Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=kernel_initializer)(x)
  x = InstanceNormalization(axis=-1)(x)
  x = L.LeakyReLU(0.2)(x)

  x = L.Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=kernel_initializer)(x)
  x = InstanceNormalization(axis=-1)(x)
  x = L.LeakyReLU(0.2)(x)

  x = L.Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=kernel_initializer)(x)
  x = InstanceNormalization(axis=-1)(x)
  x = L.LeakyReLU(0.2)(x)

  x = L.Conv2D(512, (4,4), padding='same', kernel_initializer=kernel_initializer)(x)
  x = InstanceNormalization(axis=-1)(x)
  x = L.LeakyReLU(0.2)(x)

  patch_out = L.Conv2D(1, (4,4), padding='same', kernel_initializer=kernel_initializer)(x)

  model = Model(input_, patch_out)

  model.compile(
      loss='mse', 
      optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
      loss_weights=[0.5] 
  )

  return model

def build_generator(input_shape, n_resnet=9):

  kernel_initializer = RandomNormal(stddev=0.02)

  input_ = L.Input(input_shape)

  x = L.Conv2D(64, (7,7), padding='same', kernel_initializer=kernel_initializer)(input_)
  x = InstanceNormalization(axis=-1)(x)
  x = L.Activation('relu')(x)

  x = L.Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=kernel_initializer)(x)
  x = InstanceNormalization(axis=-1)(x)
  x = L.Activation('relu')(x)

  x = L.Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=kernel_initializer)(x)
  x = InstanceNormalization(axis=-1)(x)
  x = L.Activation('relu')(x)

  for _ in range(n_resnet):
    x = resnet_block(256, x)

  x = L.Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=kernel_initializer)(x)
  x = InstanceNormalization(axis=-1)(x)
  x = L.Activation('relu')(x)

  x = L.Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=kernel_initializer)(x)
  x = InstanceNormalization(axis=-1)(x)
  x = L.Activation('relu')(x)

  x = L.Conv2D(1, (7, 7), padding='same', kernel_initializer=kernel_initializer)(x)
  x = InstanceNormalization(axis=-1)(x)
  output = L.Activation('tanh')(x)

  model = Model(input_, output)

  return model

def resnet_block(n_filters, input_layer):

  kernel_initializer = RandomNormal(stddev=0.02)

  x = L.Conv2D(n_filters, (3,3), padding='same', kernel_initializer=kernel_initializer,)(input_layer)
  x = InstanceNormalization(axis=-1)(x)
  x = L.Activation('relu')(x)

  x = L.Conv2D(n_filters, (3,3), padding='same', kernel_initializer=kernel_initializer)(x)
  x = InstanceNormalization(axis=-1)(x)
  x = L.Concatenate()([input_layer, x])
  
  return x

def build_composite_model(g_model_1, d_model, g_model_2, input_shape):

  g_model_1.trainable = True

  d_model.trainable = False

  g_model_2.trainable = False

  input_gen = L.Input(input_shape)
  gen1_out = g_model_1(input_gen)
  output_d = d_model(gen1_out)

  input_id = L.Input(input_shape)
  output_id = g_model_1(input_id)

  output_f = g_model_2(gen1_out)

  gen2_out = g_model_2(input_id)
  output_b = g_model_1(gen2_out)

  model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])


  model.compile(
      loss=['mse', 'mae', 'mae', 'mae'], 
      loss_weights=[1, 5, 10, 10], 
      optimizer=Adam(learning_rate=0.0002, beta_1=0.5)
      )
  
  return model