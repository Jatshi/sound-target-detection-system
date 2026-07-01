from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import torchvision.ops as ops

TARGET_SR = 44100
TARGET_SAMPLES = TARGET_SR
LATENT_DIM = 64
ODE_STEPS = 10
RESIDUAL_GAMMA = 1.5


class DeformableConv2d(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.offset = nn.Conv2d(in_c, 2 * k * k, k, s, p)
        self.mask = nn.Conv2d(in_c, k * k, k, s, p)
        self.weight = nn.Parameter(torch.empty(out_c, in_c, k, k))
        self.stride, self.padding = s, p
        nn.init.kaiming_uniform_(self.weight, a=1)
        nn.init.constant_(self.offset.weight, 0)
        nn.init.constant_(self.mask.weight, 0)

    def forward(self, x):
        return ops.deform_conv2d(
            x, self.offset(x), self.weight, None, self.stride, self.padding, mask=torch.sigmoid(self.mask(x))
        )


class DeformableNet(nn.Module):
    def __init__(self, num_classes: int = 5, out_dim: int = 527):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 64, 7, 2, 3), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(3, 2, 1))
        self.l1 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.l2 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.l3 = nn.Sequential(DeformableConv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.l4 = nn.Sequential(DeformableConv2d(256, 512, 3, 2, 1), nn.BatchNorm2d(512), nn.ReLU())
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.time_proj = nn.Conv1d(512, out_dim, 1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(out_dim)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.l4(self.l3(self.l2(self.l1(x))))
        x = self.freq_pool(x).squeeze(2)
        feat_time = self.time_proj(x)
        feat_global = self.global_pool(feat_time).squeeze(2)
        return self.fc(self.norm(feat_global)), feat_global, feat_time


class ResBlock2D(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride, bias=False), nn.BatchNorm2d(out_c))

    def forward(self, x):
        return F.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))


class ResNet10_TimeAware(nn.Module):
    def __init__(self, num_classes: int = 5, out_dim: int = 527):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = ResBlock2D(64, 64, 1)
        self.layer2 = ResBlock2D(64, 128, 2)
        self.layer3 = ResBlock2D(128, 256, 2)
        self.layer4 = ResBlock2D(256, 512, 2)
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.time_proj = nn.Conv1d(512, out_dim, 1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(out_dim)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.freq_pool(x).squeeze(2)
        feat_time = self.time_proj(x)
        feat_global = self.global_pool(feat_time).squeeze(2)
        return self.fc(self.norm(feat_global)), feat_global, feat_time


class SpatialAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1))) * x


class DilatedNet(nn.Module):
    def __init__(self, num_classes: int = 5, out_dim: int = 527):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, 64, 7, 2, 3), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(3, 2, 1))
        self.l1 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.l2 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 2, dilation=2), nn.BatchNorm2d(128), nn.ReLU())
        self.l3 = nn.Sequential(nn.Conv2d(128, 256, 3, 2, 4, dilation=4), nn.BatchNorm2d(256), nn.ReLU())
        self.l4 = nn.Sequential(nn.Conv2d(256, 512, 3, 2, 8, dilation=8), nn.BatchNorm2d(512), nn.ReLU())
        self.attn = SpatialAttn()
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))
        self.time_proj = nn.Conv1d(512, out_dim, 1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(out_dim)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.l4(self.l3(self.l2(self.l1(x))))
        x = self.attn(x)
        x = self.freq_pool(x).squeeze(2)
        feat_time = self.time_proj(x)
        feat_global = self.global_pool(feat_time).squeeze(2)
        return self.fc(self.norm(feat_global)), feat_global, feat_time


class AudioBaselineWrapper(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        out = self.net(x)
        return out[0] if isinstance(out, (tuple, list)) else out


class ResBlock1D(nn.Module):
    def __init__(self, c: int, k: int = 3, p: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(c, c, k, padding=p)
        self.bn1 = nn.BatchNorm1d(c)
        self.act = nn.LeakyReLU(0.2, True)
        self.conv2 = nn.Conv1d(c, c, k, padding=p)
        self.bn2 = nn.BatchNorm1d(c)

    def forward(self, x):
        return self.act(self.bn2(self.conv2(self.act(self.bn1(self.conv1(x))))) + x)


class EEG_VAE_Light(nn.Module):
    def __init__(self, time_points: int = 200, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(60, 64, 4, 2, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            ResBlock1D(64),
            nn.Conv1d(64, 128, 4, 2, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            ResBlock1D(128),
            nn.Conv1d(128, 256, 4, 2, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
        )
        self.flat_dim = 256 * (time_points // 8)
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flat_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, 4, 2, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            ResBlock1D(128),
            nn.ConvTranspose1d(128, 64, 4, 2, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            ResBlock1D(64),
            nn.ConvTranspose1d(64, 60, 4, 2, 1),
        )
        self.time = time_points

    def encode(self, x):
        h = self.encoder(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        return self.decoder(self.fc_dec(z).view(z.size(0), 256, self.time // 8))


class TimeAwareVectorField(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM, cond_dim: int = 527):
        super().__init__()
        self.cond_conv = nn.Sequential(
            nn.Conv1d(cond_dim, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.time_mlp = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64))
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 64 + 256, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, latent_dim)
        )

    def forward(self, x, t, c):
        return self.net(torch.cat([x, self.time_mlp(t), self.cond_conv(c).squeeze(2)], dim=1))


class EEGNet_Interpreter(nn.Module):
    def __init__(self, Chans: int = 60):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, (Chans, 1), groups=8, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(0.25)
        self.conv3 = nn.Conv2d(16, 16, (1, 16), padding=(0, 8), groups=16, bias=False)
        self.conv4 = nn.Conv2d(16, 16, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(16)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(0.25)
        self.fc = nn.Linear(16 * 6, 64)

    def forward_seq(self, x):
        x = x.unsqueeze(1)
        x = self.drop1(self.pool1(self.elu1(self.bn2(self.conv2(self.bn1(self.conv1(x)))))))
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.pool2(x)
        return self.drop2(x).squeeze(2)

    def forward(self, x):
        x = self.forward_seq(x)
        return self.fc(x.view(x.size(0), -1))


class BaseNeuroCAP(nn.Module):
    def __init__(self, num_classes: int = 5, cond_dim: int = 527, backbone: str = "deformable"):
        super().__init__()
        if backbone == "deformable":
            self.sound_encoder = DeformableNet(num_classes, cond_dim)
        elif backbone == "resnet10":
            self.sound_encoder = ResNet10_TimeAware(num_classes, cond_dim)
        elif backbone == "dilated":
            self.sound_encoder = DilatedNet(num_classes, cond_dim)
        else:
            raise ValueError(f"Unknown NeuroCAP backbone: {backbone}")
        self.flow_net = TimeAwareVectorField(latent_dim=LATENT_DIM, cond_dim=cond_dim)
        self.vae = EEG_VAE_Light(time_points=200, latent_dim=LATENT_DIM)
        self.interpreter = EEGNet_Interpreter(Chans=60)
        self.neuro_head = nn.Linear(64, num_classes)
        self.sound_head = nn.Sequential(nn.Dropout(0.5), nn.Linear(cond_dim, num_classes))
        self.gate_fc = nn.Sequential(nn.Linear(cond_dim + 64, 128), nn.ReLU(), nn.Linear(128, 1))
        self.classifier = nn.Sequential(nn.Linear(cond_dim + 64, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes))

    def generate_latent_and_eeg(self, cond_seq):
        b = cond_seq.shape[0]
        curr_z = torch.randn(b, LATENT_DIM, device=cond_seq.device)
        dt = 1.0 / ODE_STEPS
        for i in range(ODE_STEPS):
            t = torch.ones(b, 1, device=cond_seq.device) * (i * dt)
            curr_z = curr_z + self.flow_net(curr_z, t, cond_seq) * dt
        return curr_z, self.vae.decode(curr_z)


class QARNFusionModule(nn.Module):
    def __init__(
        self, d_audio: int = 527, d_neuro_seq: int = 16, d_neuro_global: int = 64, d_model: int = 256, num_classes: int = 5, gamma: float = 1.5
    ):
        super().__init__()
        self.audio_proj = nn.Linear(d_audio, d_model)
        self.neuro_proj = nn.Linear(d_neuro_seq, d_model)
        self.neuro_global_proj = nn.Linear(d_neuro_global, d_model)
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5
        self.quality_head = nn.Sequential(nn.Linear(d_model * 3, 128), nn.ReLU(), nn.Linear(128, 1))
        self.gate_head = nn.Sequential(nn.Linear(d_model * 3, 128), nn.ReLU(), nn.Linear(128, 1))
        self.residual_head = nn.Sequential(
            nn.LayerNorm(d_model * 3), nn.Linear(d_model * 3, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes)
        )
        self.gamma = gamma

    def forward(self, audio_seq, neuro_seq, audio_global, neuro_global, sound_logits, neuro_mode: str = "full"):
        a_seq = audio_seq.permute(0, 2, 1)
        n_seq = neuro_seq.permute(0, 2, 1)
        if neuro_mode == "zero":
            n_seq = torch.zeros_like(n_seq)
            neuro_global = torch.zeros_like(neuro_global)
        elif neuro_mode == "shuffle" and n_seq.size(0) > 1:
            idx = torch.randperm(n_seq.size(0), device=n_seq.device)
            n_seq = n_seq[idx]
            neuro_global = neuro_global[idx]
        a = self.audio_proj(a_seq)
        n = self.neuro_proj(n_seq)
        ng = self.neuro_global_proj(neuro_global)
        q, k, v = self.q(a), self.k(n), self.v(n)
        attn = F.softmax(torch.bmm(q, k.transpose(1, 2)) * self.scale, dim=-1)
        retrieved = torch.bmm(attn, v)
        conf = attn.max(dim=-1, keepdim=True).values
        a_pool = a.mean(dim=1)
        r_pool = retrieved.mean(dim=1)
        n_pool = n.mean(dim=1)
        fusion_ctx = torch.cat([a_pool, r_pool, ng], dim=1)
        quality_logits = self.quality_head(torch.cat([a_pool, n_pool, torch.abs(a_pool - n_pool)], dim=1))
        gate_logits = self.gate_head(fusion_ctx)
        residual_logits = self.residual_head(fusion_ctx)
        eff_gate = torch.sigmoid(gate_logits) * torch.sigmoid(quality_logits) * torch.clamp(0.5 + conf.mean(dim=1), max=1.0)
        main_logits = sound_logits + self.gamma * eff_gate * residual_logits
        align_loss = 1.0 - F.cosine_similarity(a_pool, r_pool, dim=1).mean()
        return main_logits, residual_logits, gate_logits, quality_logits, eff_gate, a_pool, n_pool, align_loss


class NeuroCAPQARN(BaseNeuroCAP):
    def __init__(self, num_classes: int = 5, cond_dim: int = 527, backbone: str = "deformable"):
        super().__init__(num_classes=num_classes, cond_dim=cond_dim, backbone=backbone)
        self.qarn_fusion = QARNFusionModule(d_audio=cond_dim, d_neuro_seq=16, d_neuro_global=64, num_classes=num_classes, gamma=RESIDUAL_GAMMA)

    def interpret_seq_and_global(self, eeg):
        x = eeg.unsqueeze(1)
        i = self.interpreter
        x = i.drop1(i.pool1(i.elu1(i.bn2(i.conv2(i.bn1(i.conv1(x)))))))
        x = i.conv3(x)
        x = i.conv4(x)
        x = i.bn3(x)
        x = i.elu2(x)
        x = i.pool2(x)
        x = i.drop2(x)
        return x.squeeze(2), i.fc(x.view(x.size(0), -1))

    def forward(self, spec, real_eeg=None, return_gate: bool = False, return_feat: bool = False, neuro_mode: str = "full"):
        _, feat_global, feat_time = self.sound_encoder(spec)
        z_gen, virtual_eeg = self.generate_latent_and_eeg(feat_time)
        lat_loss = torch.tensor(0.0, device=spec.device)
        if real_eeg is not None:
            with torch.no_grad():
                z_real, _ = self.vae.encode(real_eeg)
            lat_loss = F.mse_loss(z_gen, z_real)
        feat_n_seq, feat_cog_64 = self.interpret_seq_and_global(virtual_eeg)
        neuro_logits = self.neuro_head(feat_cog_64)
        sound_logits = self.sound_head(feat_global)
        main_logits, residual_logits, gate_logits, quality_logits, eff_gate, a_emb, n_emb, align_loss = self.qarn_fusion(
            feat_time, feat_n_seq, feat_global, feat_cog_64, sound_logits, neuro_mode=neuro_mode
        )
        aux = {"quality_logits": quality_logits, "residual_logits": residual_logits, "audio_emb": a_emb, "neuro_emb": n_emb, "align_loss": align_loss}
        if return_gate:
            if return_feat:
                return main_logits, sound_logits, neuro_logits, gate_logits, lat_loss, feat_global, eff_gate, feat_cog_64, aux
            return main_logits, sound_logits, neuro_logits, gate_logits, lat_loss, feat_global, eff_gate, aux
        if return_feat:
            return main_logits, sound_logits, neuro_logits, gate_logits, lat_loss, feat_global, feat_cog_64, aux
        return main_logits, sound_logits, neuro_logits, gate_logits, lat_loss, feat_global


class WaveToSpec(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = T.MelSpectrogram(sample_rate=TARGET_SR, n_fft=1024, hop_length=512, n_mels=64, power=2.0)

    def forward(self, wav):
        spec = self.mel(wav)
        spec = 10 * torch.log10(spec + 1e-9)
        spec = (spec - (-40)) / 40
        mean = spec.mean(dim=(-2, -1), keepdim=True)
        std = spec.std(dim=(-2, -1), keepdim=True)
        spec = (spec - mean) / (std + 1e-6)
        return spec.unsqueeze(1) if spec.dim() == 3 else spec
