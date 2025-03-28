import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import numpy as np
import cv2
import os
import glob
from tqdm import tqdm

# CycleGAN mimarisi kullanan bir anime dönüştürme modeli

def build_generator():
    """
    U-Net yapısında bir jeneratör oluşturur.
    """
    def downsample(filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)
        
        result = tf.keras.Sequential()
        result.add(layers.Conv2D(filters, size, strides=2, padding='same',
                                 kernel_initializer=initializer, use_bias=False))
        
        if apply_batchnorm:
            result.add(layers.BatchNormalization())
            
        result.add(layers.LeakyReLU())
        
        return result
    
    def upsample(filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)
        
        result = tf.keras.Sequential()
        result.add(layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                          kernel_initializer=initializer, use_bias=False))
        
        result.add(layers.BatchNormalization())
        
        if apply_dropout:
            result.add(layers.Dropout(0.5))
            
        result.add(layers.ReLU())
        
        return result
    
    # Giriş
    inputs = Input(shape=[256, 256, 3])
    
    # Encoder (downsampling)
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
        downsample(128, 4),  # (bs, 64, 64, 128)
        downsample(256, 4),  # (bs, 32, 32, 256)
        downsample(512, 4),  # (bs, 16, 16, 512)
        downsample(512, 4),  # (bs, 8, 8, 512)
        downsample(512, 4),  # (bs, 4, 4, 512)
        downsample(512, 4),  # (bs, 2, 2, 512)
        downsample(512, 4),  # (bs, 1, 1, 512)
    ]
    
    # Decoder (upsampling)
    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (bs, 8, 8, 1024)
        upsample(512, 4),  # (bs, 16, 16, 1024)
        upsample(256, 4),  # (bs, 32, 32, 512)
        upsample(128, 4),  # (bs, 64, 64, 256)
        upsample(64, 4),  # (bs, 128, 128, 128)
    ]
    
    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(3, 4, strides=2, padding='same',
                                  kernel_initializer=initializer,
                                  activation='tanh')  # (bs, 256, 256, 3)
    
    x = inputs
    
    # Encoder çıktılarını sakla
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    
    skips = reversed(skips[:-1])
    
    # Decoder ve skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])
    
    x = last(x)
    
    return Model(inputs=inputs, outputs=x)

def build_discriminator():
    """
    Gerçek ve üretilmiş görüntüleri ayırt etmek için bir ayrımcı model oluşturur.
    """
    initializer = tf.random_normal_initializer(0., 0.02)
    
    inp = Input(shape=[256, 256, 3], name='input_image')
    
    x = inp
    
    # Ayrımcı katmanları
    down1 = layers.Conv2D(64, 4, strides=2, padding='same',
                          kernel_initializer=initializer, use_bias=False)(x)
    leaky_relu1 = layers.LeakyReLU()(down1)
    
    down2 = layers.Conv2D(128, 4, strides=2, padding='same',
                          kernel_initializer=initializer, use_bias=False)(leaky_relu1)
    norm2 = layers.BatchNormalization()(down2)
    leaky_relu2 = layers.LeakyReLU()(norm2)
    
    down3 = layers.Conv2D(256, 4, strides=2, padding='same',
                          kernel_initializer=initializer, use_bias=False)(leaky_relu2)
    norm3 = layers.BatchNormalization()(down3)
    leaky_relu3 = layers.LeakyReLU()(norm3)
    
    # Son katman
    zero_pad1 = layers.ZeroPadding2D()(leaky_relu3)
    conv = layers.Conv2D(512, 4, strides=1,
                         kernel_initializer=initializer,
                         use_bias=False)(zero_pad1)
    norm4 = layers.BatchNormalization()(conv)
    leaky_relu4 = layers.LeakyReLU()(norm4)
    
    zero_pad2 = layers.ZeroPadding2D()(leaky_relu4)
    last = layers.Conv2D(1, 4, strides=1,
                         kernel_initializer=initializer)(zero_pad2)
    
    return Model(inputs=inp, outputs=last)

class CycleGAN(Model):
    def __init__(self):
        super(CycleGAN, self).__init__()
        self.gen_human_to_anime = build_generator()
        self.gen_anime_to_human = build_generator()
        self.disc_human = build_discriminator()
        self.disc_anime = build_discriminator()
        
        # Optimizerlar
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        
        # Kayıp fonksiyonları
        self.loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        
    def compile(self, gen_human_to_anime_optimizer, gen_anime_to_human_optimizer,
               disc_human_optimizer, disc_anime_optimizer):
        super(CycleGAN, self).compile()
        self.gen_human_to_anime_optimizer = gen_human_to_anime_optimizer
        self.gen_anime_to_human_optimizer = gen_anime_to_human_optimizer
        self.disc_human_optimizer = disc_human_optimizer
        self.disc_anime_optimizer = disc_anime_optimizer

# Eğitim işlevleri

def generator_loss(generated_output):
    """
    Jeneratör kayıp fonksiyonu
    """
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(generated_output), generated_output)

def discriminator_loss(real_output, generated_output):
    """
    Ayrımcı kayıp fonksiyonu
    """
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_output), real_output)
    generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(generated_output), generated_output)
    total_loss = real_loss + generated_loss
    return total_loss * 0.5

def cycle_consistency_loss(real_image, cycled_image):
    """
    Cycle consistency kayıp fonksiyonu
    """
    return tf.reduce_mean(tf.abs(real_image - cycled_image))

def preprocess_image(image_path, target_size=(256, 256)):
    """
    Görüntüyü ön işleme
    """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, target_size)
    img = (img / 127.5) - 1  # [-1, 1] aralığına normalize et
    return img

def create_dataset(human_images_path, anime_images_path, batch_size=1):
    """
    Eğitim veri kümesini oluşturur
    """
    human_images = glob.glob(os.path.join(human_images_path, '*.jpg'))
    anime_images = glob.glob(os.path.join(anime_images_path, '*.jpg'))
    
    human_dataset = tf.data.Dataset.from_tensor_slices(human_images)
    human_dataset = human_dataset.map(lambda x: preprocess_image(x))
    human_dataset = human_dataset.shuffle(buffer_size=len(human_images))
    human_dataset = human_dataset.batch(batch_size)
    
    anime_dataset = tf.data.Dataset.from_tensor_slices(anime_images)
    anime_dataset = anime_dataset.map(lambda x: preprocess_image(x))
    anime_dataset = anime_dataset.shuffle(buffer_size=len(anime_images))
    anime_dataset = anime_dataset.batch(batch_size)
    
    return tf.data.Dataset.zip((human_dataset, anime_dataset))

def train(model, dataset, epochs=50):
    """
    Model eğitimi
    """
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (human_batch, anime_batch) in enumerate(tqdm(dataset)):
            # Eğitim adımı
            with tf.GradientTape(persistent=True) as tape:
                # İnsan -> Anime -> İnsan
                fake_anime = model.gen_human_to_anime(human_batch, training=True)
                cycled_human = model.gen_anime_to_human(fake_anime, training=True)
                
                # Anime -> İnsan -> Anime
                fake_human = model.gen_anime_to_human(anime_batch, training=True)
                cycled_anime = model.gen_human_to_anime(fake_human, training=True)
                
                # Ayrımcı çıktıları
                disc_real_human = model.disc_human(human_batch, training=True)
                disc_fake_human = model.disc_human(fake_human, training=True)
                
                disc_real_anime = model.disc_anime(anime_batch, training=True)
                disc_fake_anime = model.disc_anime(fake_anime, training=True)
                
                # Generator kayıpları
                gen_human_to_anime_loss = generator_loss(disc_fake_anime)
                gen_anime_to_human_loss = generator_loss(disc_fake_human)
                
                # Cycle kayıpları
                cycle_human_loss = cycle_consistency_loss(human_batch, cycled_human)
                cycle_anime_loss = cycle_consistency_loss(anime_batch, cycled_anime)
                
                # Toplam generator kayıpları
                total_gen_human_to_anime_loss = gen_human_to_anime_loss + 10 * cycle_human_loss
                total_gen_anime_to_human_loss = gen_anime_to_human_loss + 10 * cycle_anime_loss
                
                # Discriminator kayıpları
                disc_human_loss = discriminator_loss(disc_real_human, disc_fake_human)
                disc_anime_loss = discriminator_loss(disc_real_anime, disc_fake_anime)
            
            # Gradientleri hesapla
            gen_human_to_anime_gradients = tape.gradient(total_gen_human_to_anime_loss,
                                                        model.gen_human_to_anime.trainable_variables)
            gen_anime_to_human_gradients = tape.gradient(total_gen_anime_to_human_loss,
                                                        model.gen_anime_to_human.trainable_variables)
            
            disc_human_gradients = tape.gradient(disc_human_loss,
                                               model.disc_human.trainable_variables)
            disc_anime_gradients = tape.gradient(disc_anime_loss,
                                               model.disc_anime.trainable_variables)
            
            # Optimize et
            model.gen_human_to_anime_optimizer.apply_gradients(
                zip(gen_human_to_anime_gradients, model.gen_human_to_anime.trainable_variables))
            model.gen_anime_to_human_optimizer.apply_gradients(
                zip(gen_anime_to_human_gradients, model.gen_anime_to_human.trainable_variables))
            
            model.disc_human_optimizer.apply_gradients(
                zip(disc_human_gradients, model.disc_human.trainable_variables))
            model.disc_anime_optimizer.apply_gradients(
                zip(disc_anime_gradients, model.disc_anime.trainable_variables))