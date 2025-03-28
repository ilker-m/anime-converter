import os
import argparse
import tensorflow as tf
from model import CycleGAN, create_dataset
import matplotlib.pyplot as plt
import time
import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='CycleGAN Anime Dönüştürücü Eğitimi')
    parser.add_argument('--human_dir', type=str, required=True, help='İnsan görüntülerinin bulunduğu dizin')
    parser.add_argument('--anime_dir', type=str, required=True, help='Anime görüntülerinin bulunduğu dizin')
    parser.add_argument('--epochs', type=int, default=50, help='Eğitim epoch sayısı')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch boyutu')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/', help='Checkpoint kaydetme dizini')
    parser.add_argument('--restore', action='store_true', help='Eğitimi devam ettirmek için checkpoint yükle')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Checkpoint dizinini oluştur
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Dataset oluştur
    dataset = create_dataset(args.human_dir, args.anime_dir, args.batch_size)
    
    # Örnek görüntüleri kaydet
    for human_batch, anime_batch in dataset.take(1):
        sample_human = human_batch[0].numpy()
        sample_anime = anime_batch[0].numpy()
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Örnek İnsan Görüntüsü')
        plt.imshow((sample_human + 1) * 0.5)  # [-1, 1] -> [0, 1]
        
        plt.subplot(1, 2, 2)
        plt.title('Örnek Anime Görüntüsü')
        plt.imshow((sample_anime + 1) * 0.5)  # [-1, 1] -> [0, 1]
        
        plt.savefig('sample_images.png')
        plt.close()
    
    # Model oluştur
    model = CycleGAN()
    
    # Optimizerları oluştur
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    
    # Model derleme
    model.compile(
        gen_human_to_anime_optimizer=generator_optimizer,
        gen_anime_to_human_optimizer=generator_optimizer,
        disc_human_optimizer=discriminator_optimizer,
        disc_anime_optimizer=discriminator_optimizer
    )
    
    # Checkpoint
    checkpoint_prefix = os.path.join(args.checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        gen_human_to_anime=model.gen_human_to_anime,
        gen_anime_to_human=model.gen_anime_to_human,
        disc_human=model.disc_human,
        disc_anime=model.disc_anime,
        gen_human_to_anime_optimizer=generator_optimizer,
        gen_anime_to_human_optimizer=generator_optimizer,
        disc_human_optimizer=discriminator_optimizer,
        disc_anime_optimizer=discriminator_optimizer
    )
    
    # Checkpoint manager
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, args.checkpoint_dir, max_to_keep=5)
    
    # Checkpoint'ten devam et
    if args.restore:
        if checkpoint_manager.latest_checkpoint:
            status = checkpoint.restore(checkpoint_manager.latest_checkpoint)
            print(f"Checkpoint'ten devam ediliyor: {checkpoint_manager.latest_checkpoint}")
        else:
            print("Uyarı: Checkpoint bulunamadı, eğitime sıfırdan başlanıyor.")
    
    # Metrikleri izle
    summary_writer = tf.summary.create_file_writer(
        os.path.join('logs/', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    )
    
    # Eğitim adımı
    @tf.function
    def train_step(human_batch, anime_batch):
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
        generator_optimizer.apply_gradients(
            zip(gen_human_to_anime_gradients, model.gen_human_to_anime.trainable_variables))
        generator_optimizer.apply_gradients(
            zip(gen_anime_to_human_gradients, model.gen_anime_to_human.trainable_variables))
        
        discriminator_optimizer.apply_gradients(
            zip(disc_human_gradients, model.disc_human.trainable_variables))
        discriminator_optimizer.apply_gradients(
            zip(disc_anime_gradients, model.disc_anime.trainable_variables))
        
        return {
            'gen_human_to_anime_loss': gen_human_to_anime_loss,
            'gen_anime_to_human_loss': gen_anime_to_human_loss,
            'cycle_human_loss': cycle_human_loss,
            'cycle_anime_loss': cycle_anime_loss,
            'disc_human_loss': disc_human_loss,
            'disc_anime_loss': disc_anime_loss
        }
    
    # Kayıp fonksiyonları
    def generator_loss(generated_output):
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(generated_output), generated_output)
    
    def discriminator_loss(real_output, generated_output):
        real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_output), real_output)
        generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(generated_output), generated_output)
        total_loss = real_loss + generated_loss
        return total_loss * 0.5
    
    def cycle_consistency_loss(real_image, cycled_image):
        return tf.reduce_mean(tf.abs(real_image - cycled_image))
    
    # Örnek görüntü üretimi
    def generate_sample(epoch, human_batch, anime_batch):
        fake_anime = model.gen_human_to_anime(human_batch, training=False)
        fake_human = model.gen_anime_to_human(anime_batch, training=False)
        
        cycled_human = model.gen_anime_to_human(fake_anime, training=False)
        cycled_anime = model.gen_human_to_anime(fake_human, training=False)
        
        plt.figure(figsize=(15, 10))
        
        display_list = [
            human_batch[0], fake_anime[0], cycled_human[0],
            anime_batch[0], fake_human[0], cycled_anime[0]
        ]
        title_list = [
            'İnsan Görüntüsü', 'Sahte Anime', 'Yeniden Oluşturulan İnsan',
            'Anime Görüntüsü', 'Sahte İnsan', 'Yeniden Oluşturulan Anime'
        ]
        
        for i in range(6):
            plt.subplot(2, 3, i+1)
            plt.title(title_list[i])
            plt.imshow((display_list[i].numpy() + 1) * 0.5)  # [-1, 1] -> [0, 1]
            plt.axis('off')
        
        plt.savefig(f'samples/epoch_{epoch+1}.png')
        plt.close()
    
    # Örnek görüntülerin kaydedileceği dizini oluştur
    os.makedirs('samples', exist_ok=True)
    
    # Eğitim
    print(f"Eğitim başlıyor... Epoch sayısı: {args.epochs}")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # İlerleme çubuğu için hazırlık
        batch_count = tf.data.experimental.cardinality(dataset).numpy()
        
        losses = {'gen_human_to_anime_loss': 0, 'gen_anime_to_human_loss': 0,
                 'cycle_human_loss': 0, 'cycle_anime_loss': 0,
                 'disc_human_loss': 0, 'disc_anime_loss': 0}
        
        for batch_idx, (human_batch, anime_batch) in enumerate(dataset):
            batch_losses = train_step(human_batch, anime_batch)
            
            # Kayıpları topla
            for k, v in batch_losses.items():
                losses[k] += v
            
            # İlerleme durumunu göster
            print(f"\rBatch {batch_idx+1}/{batch_count}", end='')
        
        print()  # Yeni satır
        
        # Ortalama kayıpları hesapla
        for k in losses.keys():
            losses[k] /= batch_count
        
        # TensorBoard kayıtları
        with summary_writer.as_default():
            for k, v in losses.items():
                tf.summary.scalar(k, v, step=epoch)
        
        # Örnek görüntüler üret
        for human_batch, anime_batch in dataset.take(1):
            generate_sample(epoch, human_batch, anime_batch)
        
        # Checkpoint kaydet
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            checkpoint_manager.save()
            print(f"Checkpoint kaydedildi: {checkpoint_manager.latest_checkpoint}")
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1} tamamlandı! Süre: {epoch_time:.2f} saniye")
        print(f"Kayıplar: {', '.join([f'{k}: {v:.4f}' for k, v in losses.items()])}")
    
    # Model kaydet
    model.gen_human_to_anime.save('models/gen_human_to_anime.h5')
    model.gen_anime_to_human.save('models/gen_anime_to_human.h5')
    
    print(f"Eğitim tamamlandı! Toplam süre: {(time.time() - start_time) / 60:.2f} dakika")

if __name__ == "__main__":
    main()
