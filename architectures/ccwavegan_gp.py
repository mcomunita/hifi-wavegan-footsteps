# Class-Conditional WaveGAN Generator + 
# Class-conditional WaveGAN Discriminator
"""
Paradigm: Wassertein GAN with Gradient Penalty (WGAN-GP)
"""

import os
import time
import torch
import shutil

from utils import gradients_status
from utils.utils_architectures import synth_samples_at_batch, synth_fad_samples_at_batch, synth_mmd_samples_at_batch, \
                                save_wavegan_at_batch, \
                                compute_fad_at_batch, compute_mmd_at_batch

class CCWaveGAN_GP(torch.nn.Module):

    def __init__(
        self,
        latent_dim,
        generator,
        discriminator,
        n_classes,
        g_optimizer,
        d_optimizer,
        device,
        d_extra_steps,
        gp_weight=10.0,
    ):
        super(CCWaveGAN_GP, self).__init__()
        self.latent_dim = latent_dim
        self.generator = generator
        self.discriminator = discriminator
        self.n_classes = n_classes
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.device = device
        self.d_steps = d_extra_steps
        self.gp_weight = gp_weight
        self.g_loss_fn = self.generator_loss
        self.d_loss_fn = self.discriminator_loss

        self.discriminator.to(self.device)
        self.generator.to(self.device)

    def apply_zero_grad(self):
        self.generator.zero_grad()
        self.g_optimizer.zero_grad()

        self.discriminator.zero_grad()
        self.d_optimizer.zero_grad()
        
    def enable_disc_disable_gen(self):
        gradients_status(self.discriminator, True)
        gradients_status(self.generator, False)

    def enable_gen_disable_disc(self):
        gradients_status(self.discriminator, False)
        gradients_status(self.generator, True)

    def disable_all(self):
        gradients_status(self.discriminator, False)
        gradients_status(self.generator, False)

    def discriminator_loss(self, real_samples, fake_samples):
        real_loss = torch.mean(real_samples)
        fake_loss = torch.mean(fake_samples)
        return fake_loss - real_loss
    
    def generator_loss(self, fake_samples):
        return -torch.mean(fake_samples)

    def gradient_penalty(self, batch_size, real_samples, fake_samples, labels):
        """ Calculates the gradient penalty.
        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # get the interpolated image
        alpha = torch.normal(mean=0.0, std=1.0, size=[batch_size, 1, 1]).to(self.device)
        diff = fake_samples - real_samples
        interpolated = real_samples + alpha * diff
        interpolated = torch.autograd.Variable(interpolated, requires_grad=True)
        
        # 1. Get the discriminator output for this interpolated image.
        pred = self.discriminator(interpolated, labels)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        ones = torch.ones(pred.size()).to(self.device)

        gradients = torch.autograd.grad(
            outputs=pred,
            inputs=interpolated,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        # calculate gradient penalty
        grad_penalty = \
            ((gradients.view(gradients.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()
        assert not (torch.isnan(grad_penalty))
        return grad_penalty
        
    def train_batch(self, x, y, batch_size):
        # get random indexes for the batch
        idx = torch.randint(low=0, high=x.shape[0], size=[batch_size])
        
        # get real samples and labels
        real_samples = x[idx].to(self.device)
        labels = y[idx].to(self.device)
        
        # For each batch, we are going to perform the
        # following steps as laid out in the original paper.
        # 1. Train discriminator for N steps (typ. 5):
        #       - compute discriminator loss
        #       - compute gradient penalty
        #       - add weighted gradient penalty to loss
        # 2. Train the generator for 1 step
        
        # -- Discriminator
        self.enable_disc_disable_gen()

        for i in range(self.d_steps):    
            
            # get random latent vecs
            random_latent_vectors = torch.normal(
                mean=0.0, 
                std=1.0, 
                size=[batch_size, self.latent_dim]
            ).to(self.device)

            # generate fake samples
            fake_samples = self.generator(random_latent_vectors, labels)
            
            # get logits for fake samples and real samples
            fake_logits = self.discriminator(fake_samples, labels)
            real_logits = self.discriminator(real_samples, labels)
            
            # discriminator loss
            d_cost = self.d_loss_fn(real_samples=real_logits, fake_samples=fake_logits)
            
            # gradient penalty
            gp = self.gradient_penalty(batch_size, real_samples, fake_samples, labels) 

            # total loss
            d_loss = d_cost + gp * self.gp_weight

            # update discriminator
            self.apply_zero_grad()
            assert not (torch.isnan(d_loss))
            d_loss.backward()
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
        generated_samples = self.generator(random_latent_vectors, labels)
        
        # logits for fake samples
        generated_samples_logits = self.discriminator(generated_samples, labels)
        
        # generator loss
        g_loss = self.g_loss_fn(generated_samples_logits)
        
        # gen gradient update
        self.apply_zero_grad()
        g_loss.backward()
        self.g_optimizer.step()
        
        self.disable_all()
        
        return d_loss, g_loss
    
    def train(self, 
            x, 
            y, 
            batch_size, 
            n_batches, 
            synth_frequency,
            n_synth_samples,  
            save_frequency,
            metrics_frequency,
            n_synth_samples_metrics,
            sr, 
            n_classes, 
            checkpoints_path, 
            override_saved_model,
            writer,
            audio_path,
            verbose
    ):  
        self.generator.train()
        self.discriminator.train()

        d_writer = writer[0]
        g_writer = writer[1]

        for batch in range(n_batches):
            start_time = time.time()
            d_loss, g_loss = self.train_batch(x, y, batch_size)
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
                    sr=sr
                )
            
            # save model every n batches
            if batch % save_frequency == 0:
                save_wavegan_at_batch(
                    generator=self.generator, 
                    discriminator=self.discriminator,
                    g_optimizer=self.g_optimizer, 
                    d_optimizer=self.d_optimizer,
                    g_loss=g_loss,
                    d_loss=d_loss,  
                    batch_number=batch, 
                    checkpoints_path=checkpoints_path, 
                    override_saved_model=override_saved_model
                )
            
            # compute fad every n batches
            if batch % metrics_frequency == 0:
                output_dir = 'fad_synth_audio'
                
                # synth temporary samples for fad computation
                synth_fad_samples_at_batch(
                    generator=self.generator, 
                    batch_number=batch, 
                    checkpoints_path=checkpoints_path, 
                    output_dir=output_dir, 
                    n_classes=n_classes, 
                    n_samples_per_class=n_synth_samples_metrics, 
                    latent_dim=self.latent_dim, 
                    sr=sr
                )

                fad = compute_fad_at_batch(
                        real_samples_path=os.path.abspath(audio_path),
                        synth_samples_path=os.path.join(checkpoints_path, output_dir),
                        checkpoints_path=checkpoints_path,
                        batch_number=batch,
                        verbose=verbose,
                    )
                
                # delete temporary samples
                shutil.rmtree(os.path.join(checkpoints_path, output_dir))

                # Tensorboard
                g_writer.add_scalar("VGGish FAD", fad, batch)
                
            # compute mmd every n batches
            if batch % metrics_frequency == 0:
                output_dir = 'mmd_synth_audio'
                
                # synth temporary samples for fad computation
                synth_mmd_samples_at_batch(
                    generator=self.generator, 
                    batch_number=batch, 
                    checkpoints_path=checkpoints_path, 
                    output_dir=output_dir, 
                    n_classes=n_classes, 
                    n_samples_per_class=n_synth_samples_metrics, 
                    latent_dim=self.latent_dim, 
                    sr=sr
                )

                mmd = compute_mmd_at_batch(
                        real_samples_path=os.path.abspath(audio_path),
                        synth_samples_path=os.path.join(checkpoints_path, output_dir),
                        checkpoints_path=checkpoints_path,
                        batch_number=batch,
                        verbose=verbose,
                    )
                
                # delete temporary samples
                shutil.rmtree(os.path.join(checkpoints_path, output_dir))

                # Tensorboard
                g_writer.add_scalar("OpenL3 MMD", mmd, batch)