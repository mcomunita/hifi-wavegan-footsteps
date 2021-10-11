# Class-Conditional WaveGAN Generator + 
# Class-conditional HiFi-GAN Discriminator
"""
Paradigm: Least-squares GAN (LS-GAN)
"""

import os
import time
import torch
import shutil
from utils.utils_architectures import gradients_status
from utils.utils_training import synth_samples_at_batch, synth_fad_samples_at_batch, synth_mmd_samples_at_batch, \
                                save_wavegan_hifigan_at_batch, \
                                compute_fad_at_batch, compute_mmd_at_batch


torch.autograd.set_detect_anomaly(True)

class CCWaveGAN_CCHiFiGAN(torch.nn.Module):

    def __init__(
        self,
        latent_dim,
        generator,
        multiperiod_disc,
        multiscale_disc,
        n_classes,
        g_optimizer,
        d_optimizer,
        device,
    ):

        super(CCWaveGAN_CCHiFiGAN, self).__init__()
        self.latent_dim = latent_dim
        self.generator = generator
        self.mpd = multiperiod_disc
        self.msd = multiscale_disc
        self.n_classes = n_classes
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.device = device
        self.g_loss_fn = self.generator_loss
        self.d_loss_fn = self.discriminator_loss

        self.generator.to(self.device)
        self.mpd.to(self.device)
        self.msd.to(self.device)

    def apply_zero_grad(self):
        self.generator.zero_grad()
        self.g_optimizer.zero_grad()

        self.mpd.zero_grad()
        self.msd.zero_grad()
        self.d_optimizer.zero_grad()
        
    def enable_disc_disable_gen(self):
        gradients_status(self.mpd, True)
        gradients_status(self.msd, True)
        gradients_status(self.generator, False)

    def enable_gen_disable_disc(self):
        gradients_status(self.mpd, False)
        gradients_status(self.msd, False)
        gradients_status(self.generator, True)

    def disable_all(self):
        gradients_status(self.mpd, False)
        gradients_status(self.msd, False)
        gradients_status(self.generator, False)

    def feature_loss(self, fmap_r, fmap_g):
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))
        return loss*2

    def discriminator_loss(self, disc_real_outputs, disc_generated_outputs):
        loss = 0
        r_losses = []
        g_losses = []
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean((1-dr)**2)
            g_loss = torch.mean(dg**2)
            loss += (r_loss + g_loss)
            r_losses.append(r_loss.item())
            g_losses.append(g_loss.item())

        return loss, r_losses, g_losses

    def generator_loss(self, disc_outputs):
        loss = 0
        gen_losses = []
        for dg in disc_outputs:
            l = torch.mean((1-dg)**2)
            gen_losses.append(l)
            loss += l

        return loss, gen_losses
        
    def train_batch(self, x, y, batch_size, loss_weight=1.0):
        # get random indeces for the batch
        idx = torch.randint(low=0, high=x.shape[0], size=[batch_size])
        
        # get real samples and labels
        real_samples = x[idx].to(self.device)
        labels = y[idx].to(self.device)
                    
        # -- Discriminator
        self.enable_disc_disable_gen()

        # get random latent vecs
        random_latent_vectors = torch.normal(
            mean=0.0, 
            std=1.0, 
            size=[batch_size, self.latent_dim]
        ).to(self.device)

        # generate fake samples
        y_g_hat = self.generator(random_latent_vectors, labels)

        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = self.mpd(real_samples, y_g_hat, labels, labels)
        loss_disc_f, losses_disc_f_r, losses_disc_f_g = self.d_loss_fn(y_df_hat_r, y_df_hat_g)

        # MSD
        y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(real_samples, y_g_hat, labels, labels)
        loss_disc_s, losses_disc_s_r, losses_disc_s_g = self.d_loss_fn(y_ds_hat_r, y_ds_hat_g)

        # total disc loss
        loss_disc_all = loss_disc_s + loss_disc_f

        # update discriminator
        self.apply_zero_grad()
        loss_disc_all.backward()
        self.d_optimizer.step()
        

        # -- Generator
        self.enable_gen_disable_disc()
        # get random latent vecs
        random_latent_vectors = torch.normal(
            mean=0.0, 
            std=1.0, 
            size=[batch_size, self.latent_dim]
        ).to(self.device)

        # generate fake samples
        y_g_hat = self.generator(random_latent_vectors, labels)
        
        # logits and fmaps real and fake samples through mpd
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(real_samples, y_g_hat, labels, labels)
        
        # logits and fmaps real and fake samples through msd
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(real_samples, y_g_hat, labels, labels)
        
        # feature loss
        loss_fm_f = self.feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = self.feature_loss(fmap_s_r, fmap_s_g)

        # generator loss
        loss_gen_f, losses_gen_f = self.generator_loss(y_df_hat_g)
        loss_gen_s, losses_gen_s = self.generator_loss(y_ds_hat_g)

        # total generator loss
        loss_gen_all = (loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f) * loss_weight

        # gen gradient update
        self.apply_zero_grad()
        loss_gen_all.backward()
        self.g_optimizer.step()
        
        self.disable_all()
        
        return loss_disc_all, loss_gen_all

    def train(self, 
            x, 
            y, 
            batch_size, 
            batches, 
            synth_frequency,
            n_synth_samples,  
            save_frequency,
            loss_weight_frequency,
            n_loss_weight_synth_samples,
            sampling_rate, 
            n_classes, 
            checkpoints_path, 
            override_saved_model,
            writer,
            audio_path,
            verbose
    ):
        self.generator.train()
        self.mpd.train()
        self.msd.train()

        d_writer = writer[0]
        g_writer = writer[1]

        loss_weight = 1.0

        for batch in range(batches):
            start_time = time.time()
            d_loss, g_loss = self.train_batch(x, y, batch_size, loss_weight)
            loss_weight = 1.0
            end_time = time.time()
            time_batch = (end_time - start_time)
            print(f'Batch: {batch} == Batch size: {batch_size} == Time elapsed: {time_batch:.2f} == d_loss: {d_loss:.4f}, g_loss: {g_loss:.4f}')

            # Tensorboard
            d_writer.add_scalar("D_Loss", d_loss, batch)
            d_writer.add_scalar("Loss", d_loss, batch)

            g_writer.add_scalar("G_Loss", g_loss, batch)
            g_writer.add_scalar("Loss", g_loss, batch)
            
            # synth samples every n batches
            if batch % synth_frequency == 0 :
                synth_samples_at_batch(
                    generator=self.generator, 
                    batch_number=batch, 
                    checkpoints_path=checkpoints_path, 
                    output_dir='synth_audio', 
                    n_classes=n_classes, 
                    n_samples_per_class=n_synth_samples, 
                    latent_dim=self.latent_dim, 
                    sr=sampling_rate
                )
            
            # save model every n batches
            if batch % save_frequency == 0:
                save_wavegan_hifigan_at_batch(
                    generator=self.generator, 
                    msd=self.msd, 
                    mpd=self.mpd, 
                    g_optimizer=self.g_optimizer, 
                    d_optimizer=self.d_optimizer,
                    d_loss=d_loss, g_loss=g_loss, 
                    batch_number=batch, 
                    checkpoints_path=checkpoints_path, 
                    override_saved_model=override_saved_model
                )
            
            # compute fad every n batches
            if batch % loss_weight_frequency == 0:
                output_dir = 'fad_synth_audio'
                
                # synth temporary samples for fad computation
                synth_fad_samples_at_batch(
                    generator=self.generator, 
                    batch_number=batch, 
                    checkpoints_path=checkpoints_path, 
                    output_dir=output_dir, 
                    n_classes=n_classes, 
                    n_samples_per_class=n_loss_weight_synth_samples, 
                    latent_dim=self.latent_dim, 
                    sr=sampling_rate
                )

                fad = compute_fad_at_batch(
                        real_samples_path=os.path.abspath(audio_path),
                        synth_samples_path=os.path.join(checkpoints_path, output_dir),
                        checkpoints_path=checkpoints_path,
                        batch_number=batch,
                        verbose=verbose,
                    )
                
                # to use FAD loss uncomment this
                loss_weight = min(2.0, fad * 0.5)
                print(f'Loss Weight: {loss_weight:.4f}')
                
                # delete temporary samples
                shutil.rmtree(os.path.join(checkpoints_path, output_dir))

                # Tensorboard
                g_writer.add_scalar("VGGish FAD", fad, batch)
                g_writer.add_scalar("Loss Weight", loss_weight, batch)

            # compute mmd every n batches
            if batch % loss_weight_frequency == 0:
                output_dir = 'mmd_synth_audio'
                
                # synth temporary samples for fad computation
                synth_mmd_samples_at_batch(
                    generator=self.generator, 
                    batch_number=batch, 
                    checkpoints_path=checkpoints_path, 
                    output_dir=output_dir, 
                    n_classes=n_classes, 
                    n_samples_per_class=n_loss_weight_synth_samples, 
                    latent_dim=self.latent_dim, 
                    sr=sampling_rate
                )

                mmd = compute_mmd_at_batch(
                        real_samples_path=os.path.abspath(audio_path),
                        synth_samples_path=os.path.join(checkpoints_path, output_dir),
                        checkpoints_path=checkpoints_path,
                        batch_number=batch,
                        verbose=verbose,
                    )
                
                # to use MMD loss uncomment this
                # loss_weight = min(2.0, mmd * 0.5)
                # print(f'Loss Weight: {loss_weight:.4f}')
                
                # delete temporary samples
                shutil.rmtree(os.path.join(checkpoints_path, output_dir))

                # Tensorboard
                g_writer.add_scalar("OpenL3 MMD", mmd, batch)
                # g_writer.add_scalar("Loss Weight", loss_weight, batch)

                


