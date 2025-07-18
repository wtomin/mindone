# Adapted from
# https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/opensora/models/causalvideovae/model/losses/perceptual_loss.py

import mindspore as ms
from mindspore import mint, nn, ops

from .discriminator import NLayerDiscriminator3D, weights_init
from .lpips import LPIPS


def hinge_d_loss(logits_real, logits_fake):
    loss_real = mint.mean(mint.nn.functional.relu(1.0 - logits_real))
    loss_fake = mint.mean(mint.nn.functional.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        mint.mean(mint.nn.functional.softplus(-logits_real)) + mint.mean(mint.nn.functional.softplus(logits_fake))
    )
    return d_loss


def hinge_d_loss_with_exemplar_weights(logits_real, logits_fake, weights):
    assert weights.shape[0] == logits_real.shape[0] == logits_fake.shape[0]
    loss_real = mint.mean(mint.nn.functional.relu(1.0 - logits_real), dim=[1, 2, 3])
    loss_fake = mint.mean(mint.nn.functional.relu(1.0 + logits_fake), dim=[1, 2, 3])
    loss_real = (weights * loss_real).sum() / weights.sum()
    loss_fake = (weights * loss_fake).sum() / weights.sum()
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight


def measure_perplexity(predicted_indices, n_embed):
    # src: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py
    # eval cluster perplexity. when perplexity == num_embeddings then all clusters are used exactly equally
    encodings = mint.nn.functional.one_hot(predicted_indices, n_embed).float().reshape(-1, n_embed)
    avg_probs = encodings.mean(0)
    perplexity = (-(avg_probs * mint.log(avg_probs + 1e-10)).sum()).exp()
    cluster_use = mint.sum(avg_probs > 0)
    return perplexity, cluster_use


def l1(x, y):
    return mint.abs(x - y)


def l2(x, y):
    return mint.pow((x - y), 2)


class LPIPSWithDiscriminator3D(nn.Cell):
    def __init__(
        self,
        disc_start,
        logvar_init=0.0,
        kl_weight=1.0,
        pixelloss_weight=1.0,
        perceptual_weight=1.0,
        # --- Discriminator Loss ---
        disc_num_layers=4,
        disc_in_channels=3,
        disc_factor=1.0,
        disc_weight=1.0,
        use_actnorm=False,
        disc_conditional=False,
        disc_loss="hinge",
    ):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.logvar = ms.Parameter(mint.ones(shape=[]) * logvar_init)

        self.discriminator = NLayerDiscriminator3D(
            input_nc=disc_in_channels, n_layers=disc_num_layers, use_actnorm=use_actnorm
        ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = ms.grad(nll_loss, last_layer)[0]
            g_grads = ms.grad(g_loss, last_layer)[0]
        else:
            nll_grads = ms.grad(nll_loss, self.last_layer[0])[0]
            g_grads = ms.grad(g_loss, self.last_layer[0])[0]

        d_weight = ops.norm(nll_grads) / (ops.norm(g_grads) + 1e-4)
        d_weight = mint.clamp(d_weight, 0.0, 1e4)
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def construct(
        self,
        inputs,
        reconstructions,
        posteriors,
        optimizer_idx,
        global_step,
        split="train",
        weights=None,
        last_layer=None,
        cond=None,
    ):
        # b c t h w -> (b t) c h w
        b, c, t, h, w = inputs.shape
        inputs = inputs.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
        # b c t h w -> (b t) c h w
        b, c, t, h, w = reconstructions.shape
        reconstructions = reconstructions.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
        rec_loss = mint.abs(inputs - reconstructions)
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs, reconstructions)
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        nll_loss = rec_loss / mint.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights * nll_loss
        weighted_nll_loss = mint.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = mint.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = mint.sum(kl_loss) / kl_loss.shape[0]
        # (b t) c h w -> b c t h w
        _, c, h, w = inputs.shape
        inputs = inputs.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        _, c, h, w = reconstructions.shape
        reconstructions = reconstructions.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        # GAN Part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions)
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(mint.cat((reconstructions, cond), dim=1))
            g_loss = -mint.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError as e:
                    assert not self.training, print(e)
                    d_weight = ms.Tensor(0.0)
            else:
                d_weight = ms.Tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss
            # log = {
            #     "{}/total_loss".format(split): loss.clone().mean(),
            #     "{}/logvar".format(split): self.logvar,
            #     "{}/kl_loss".format(split): kl_loss.mean(),
            #     "{}/nll_loss".format(split): nll_loss.mean(),
            #     "{}/rec_loss".format(split): rec_loss.mean(),
            #     "{}/d_weight".format(split): d_weight,
            #     "{}/disc_factor".format(split): ms.Tensor(disc_factor),
            #     "{}/g_loss".format(split): g_loss.mean(),
            # }
            return loss

        if optimizer_idx == 1:
            if cond is None:
                logits_real = self.discriminator(inputs)
                logits_fake = self.discriminator(reconstructions)
            else:
                logits_real = self.discriminator(mint.cat((inputs, cond), dim=1))
                logits_fake = self.discriminator(mint.cat((reconstructions, cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            # log = {
            #     "{}/disc_loss".format(split): d_loss.clone().mean(),
            #     "{}/logits_real".format(split): logits_real.mean(),
            #     "{}/logits_fake".format(split): logits_fake.mean(),
            # }
            return d_loss  # , log
