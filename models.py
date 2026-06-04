"""Spatial backbone + temporal head (GRU / LSTM / Transformer)."""
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        return x * self.spatial_attention(x)


def create_convnext_tiny(pretrained=True):
    from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    model = convnext_tiny(weights=weights)
    if hasattr(model, 'features') and isinstance(model.features, nn.Sequential):
        model.features.add_module('cbam', CBAM(768))
        print('[OK] CBAM attached to ConvNeXt-Tiny backbone')
    return model


class SpatialTripleEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        base = create_convnext_tiny(pretrained=pretrained)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = 768
        self.fusion = nn.Sequential(
            nn.Linear(self.feat_dim * 3, self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.person_gate = nn.Sequential(nn.Linear(self.feat_dim, 1), nn.Sigmoid())
        self.hand_gate = nn.Sequential(nn.Linear(self.feat_dim, 1), nn.Sigmoid())

    def freeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_layer4(self):
        if len(self.features) > 4:
            for p in self.features[4].parameters():
                p.requires_grad = True
        if hasattr(self.features, 'cbam'):
            for p in self.features.cbam.parameters():
                p.requires_grad = True

    def _encode_single(self, x):
        f = self.pool(self.features(x))
        return f.view(f.size(0), -1)

    def encode_frame(self, x_person, x_hand, x_wheel):
        fp = self._encode_single(x_person)
        fh = self._encode_single(x_hand)
        fw = self._encode_single(x_wheel)
        fp = fp * self.person_gate(fp)
        fh = fh * self.hand_gate(fh)
        return self.fusion(torch.cat([fp, fh, fw], dim=1))

    def forward_frames(self, person, hand, wheel):
        b, t = person.shape[0], person.shape[1]
        person = person.reshape(b * t, *person.shape[2:])
        hand = hand.reshape(b * t, *hand.shape[2:])
        wheel = wheel.reshape(b * t, *wheel.shape[2:])
        feat = self.encode_frame(person, hand, wheel)
        return feat.view(b, t, -1)


class TemporalTransformerHead(nn.Module):
    def __init__(self, feat_dim, num_classes, nhead=4, num_layers=2, dropout=0.3):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=feat_dim, nhead=nhead, dim_feedforward=feat_dim * 2,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feat_dim))
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, seq):
        b = seq.size(0)
        cls = self.cls_token.expand(b, -1, -1)
        x = self.encoder(torch.cat([cls, seq], dim=1))
        return self.classifier(x[:, 0])


class TemporalTripleModel(nn.Module):
    """5-frame clip (t-2..t+2): ConvNeXt per frame + vote / GRU / LSTM / Transformer."""
    def __init__(self, num_classes, temporal='vote', pretrained=True, hidden_dim=256, dropout=0.3,
                 frame_dropout=0.0):
        super().__init__()
        self.encoder = SpatialTripleEncoder(pretrained=pretrained)
        feat_dim = self.encoder.feat_dim
        self.temporal_type = temporal.lower()
        self.temporal = None
        self.classifier = None
        self.temporal_head = None

        if self.temporal_type == 'vote':
            pass  # mean-logit vote only; train frame_classifier
        elif self.temporal_type == 'lstm':
            self.temporal = nn.LSTM(feat_dim, hidden_dim, batch_first=True, bidirectional=True)
            self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim * 2, num_classes))
        elif self.temporal_type == 'gru':
            self.temporal = nn.GRU(feat_dim, hidden_dim, batch_first=True, bidirectional=True)
            self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim * 2, num_classes))
        elif self.temporal_type == 'transformer':
            self.temporal_head = TemporalTransformerHead(feat_dim, num_classes, dropout=dropout)
        else:
            raise ValueError(f"Unknown temporal: {temporal} (use vote, gru, lstm, transformer)")

        if frame_dropout > 0:
            self.frame_classifier = nn.Sequential(
                nn.Dropout(frame_dropout),
                nn.Linear(feat_dim, num_classes),
            )
        else:
            self.frame_classifier = nn.Linear(feat_dim, num_classes)
        self.frame_dropout = frame_dropout

    def freeze_backbone(self):
        self.encoder.freeze_backbone()

    def unfreeze_layer4(self):
        self.encoder.unfreeze_layer4()

    def center_time_index(self, seq_len):
        """Index of center frame t in clip [t-2..t+2]."""
        return seq_len // 2

    def freeze_for_temporal_training(self):
        """Freeze backbone; train only temporal head (vote: frame_classifier only)."""
        for p in self.parameters():
            p.requires_grad = False
        if self.temporal_type == 'vote':
            for p in self.frame_classifier.parameters():
                p.requires_grad = True
        elif self.temporal_type in ('gru', 'lstm'):
            for p in self.temporal.parameters():
                p.requires_grad = True
            for p in self.classifier.parameters():
                p.requires_grad = True
        elif self.temporal_type == 'transformer':
            for p in self.temporal_head.parameters():
                p.requires_grad = True

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward_frame_logits(self, person, hand, wheel):
        """Per-frame logits [B, T, num_classes] for temporal voting."""
        seq = self.encoder.forward_frames(person, hand, wheel)
        return self.frame_classifier(seq)

    def forward(self, person, hand, wheel, use_vote=False):
        seq = self.encoder.forward_frames(person, hand, wheel)
        frame_logits = self.frame_classifier(seq)

        if self.temporal_type == 'vote' or use_vote:
            return frame_logits.mean(dim=1)

        if self.temporal_type == 'transformer':
            temporal_logits = self.temporal_head(seq)
        else:
            out, _ = self.temporal(seq)
            center = self.center_time_index(out.size(1))
            temporal_logits = self.classifier(out[:, center, :])

        return temporal_logits


class SpatialFiveEncoder(nn.Module):
    """Shared ConvNeXt-Tiny on face + dual hands + wheel + mirror (fixed camera)."""

    def __init__(self, pretrained=True):
        super().__init__()
        base = create_convnext_tiny(pretrained=pretrained)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = 768
        self.fusion = nn.Sequential(
            nn.Linear(self.feat_dim * 5, self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.face_gate = nn.Sequential(nn.Linear(self.feat_dim, 1), nn.Sigmoid())
        self.left_gate = nn.Sequential(nn.Linear(self.feat_dim, 1), nn.Sigmoid())
        self.right_gate = nn.Sequential(nn.Linear(self.feat_dim, 1), nn.Sigmoid())
        self.mirror_gate = nn.Sequential(nn.Linear(self.feat_dim, 1), nn.Sigmoid())

    def freeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_layer4(self):
        if len(self.features) > 4:
            for p in self.features[4].parameters():
                p.requires_grad = True
        if hasattr(self.features, 'cbam'):
            for p in self.features.cbam.parameters():
                p.requires_grad = True

    def _encode_single(self, x):
        f = self.pool(self.features(x))
        return f.view(f.size(0), -1)

    def encode_frame(self, face, left, right, wheel, mirror):
        ff = self._encode_single(face)
        fl = self._encode_single(left)
        fr = self._encode_single(right)
        fw = self._encode_single(wheel)
        fm = self._encode_single(mirror)
        ff = ff * self.face_gate(ff)
        fl = fl * self.left_gate(fl)
        fr = fr * self.right_gate(fr)
        fm = fm * self.mirror_gate(fm)
        fused = self.fusion(torch.cat([ff, fl, fr, fw, fm], dim=1))
        hand_pair = torch.cat([fl, fr], dim=1)
        return fused, fm, hand_pair

    def forward_frames(self, face, left, right, wheel, mirror):
        b, t = face.shape[0], face.shape[1]
        face = face.reshape(b * t, *face.shape[2:])
        left = left.reshape(b * t, *left.shape[2:])
        right = right.reshape(b * t, *right.shape[2:])
        wheel = wheel.reshape(b * t, *wheel.shape[2:])
        mirror = mirror.reshape(b * t, *mirror.shape[2:])
        fused, mirror_f, hand_pair = self.encode_frame(face, left, right, wheel, mirror)
        return (
            fused.view(b, t, -1),
            mirror_f.view(b, t, -1),
            hand_pair.view(b, t, -1),
        )


class FiveROIModel(nn.Module):
    """Single-frame five-ROI classifier with mirror/phone auxiliary heads."""

    roi_layout = 'five'

    def __init__(self, num_classes, pretrained=True, dropout=0.3):
        super().__init__()
        self.encoder = SpatialFiveEncoder(pretrained=pretrained)
        d = self.encoder.feat_dim
        self.classifier = nn.Linear(d, num_classes)
        self.mirror_aux = nn.Sequential(nn.Linear(d, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))
        self.phone_aux = nn.Sequential(nn.Linear(d * 2, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))
        self.use_inference_gates = True
        self.class_names = None

    def freeze_backbone(self):
        self.encoder.freeze_backbone()

    def unfreeze_layer4(self):
        self.encoder.unfreeze_layer4()

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, face, left, right, wheel, mirror, apply_gates=None, class_names=None,
                gate_mirror_score=None, gate_phone_score=None):
        fused, mirror_f, hand_pair = self.encoder.encode_frame(face, left, right, wheel, mirror)
        logits = self.classifier(fused)
        mirror_logit = self.mirror_aux(mirror_f)
        phone_logit = self.phone_aux(hand_pair)

        do_gates = apply_gates if apply_gates is not None else (
            self.use_inference_gates and not self.training
        )
        names = class_names or self.class_names
        if do_gates and names is not None:
            from inference_gates import apply_behavior_gates, combine_gate_scores

            ms = torch.sigmoid(mirror_logit.detach()).squeeze(-1)
            ps = torch.sigmoid(phone_logit.detach()).squeeze(-1)
            if gate_mirror_score is not None:
                ms = combine_gate_scores(gate_mirror_score, ms)
            if gate_phone_score is not None:
                ps = combine_gate_scores(gate_phone_score, ps)
            logits = apply_behavior_gates(logits, names, ms, ps)

        aux = {'mirror_logit': mirror_logit, 'phone_logit': phone_logit}
        return logits, aux


class TemporalFiveModel(nn.Module):
    """
    Five-ROI temporal: Transformer / center / smooth_vote.
    Training: dual loss CE(temporal) + w * CE(center_frame) — avoids plain vote hurting left classes.
    """

    roi_layout = 'five'

    def __init__(self, num_classes, pretrained=True, dropout=0.3, frame_dropout=0.0,
                 temporal='transformer'):
        super().__init__()
        self.encoder = SpatialFiveEncoder(pretrained=pretrained)
        d = self.encoder.feat_dim
        self.temporal_backend = temporal.lower()
        fd = frame_dropout if frame_dropout > 0 else 0.0
        self.frame_classifier = nn.Sequential(
            nn.Dropout(fd),
            nn.Linear(d, num_classes),
        )
        self.temporal_head = None
        if self.temporal_backend == 'transformer':
            self.temporal_head = TemporalTransformerHead(d, num_classes, dropout=dropout)
        self.mirror_aux = nn.Sequential(nn.Linear(d, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))
        self.phone_aux = nn.Sequential(nn.Linear(d * 2, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))
        self.use_inference_gates = True
        self.class_names = None

    def freeze_backbone(self):
        self.encoder.freeze_backbone()

    def unfreeze_layer4(self):
        self.encoder.unfreeze_layer4()

    def freeze_for_temporal_training(self):
        for p in self.parameters():
            p.requires_grad = False
        train_modules = [self.frame_classifier, self.mirror_aux, self.phone_aux]
        if self.temporal_head is not None:
            train_modules.append(self.temporal_head)
        for module in train_modules:
            for p in module.parameters():
                p.requires_grad = True

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def center_time_index(self, seq_len):
        return seq_len // 2

    def _encode_sequences(self, face, left, right, wheel, mirror):
        return self.encoder.forward_frames(face, left, right, wheel, mirror)

    def forward_frame_logits(self, face, left, right, wheel, mirror):
        seq, mirror_seq, hand_seq = self._encode_sequences(face, left, right, wheel, mirror)
        return self.frame_classifier(seq), mirror_seq, hand_seq

    def _temporal_logits(self, seq, frame_logits):
        if self.temporal_backend == 'transformer' and self.temporal_head is not None:
            return self.temporal_head(seq)
        if self.temporal_backend == 'smooth_vote':
            return frame_logits.mean(dim=1)
        center = self.center_time_index(frame_logits.size(1))
        return frame_logits[:, center, :]

    def _apply_gates(self, logits, mirror_logit, phone_logit, apply_gates, class_names,
                     gate_mirror_score, gate_phone_score):
        do_gates = apply_gates if apply_gates is not None else (
            self.use_inference_gates and not self.training
        )
        names = class_names or self.class_names
        if not do_gates or names is None:
            return logits
        from inference_gates import apply_behavior_gates, combine_gate_scores

        ms = torch.sigmoid(mirror_logit.detach()).squeeze(-1)
        ps = torch.sigmoid(phone_logit.detach()).squeeze(-1)
        if gate_mirror_score is not None:
            ms = combine_gate_scores(gate_mirror_score, ms)
        if gate_phone_score is not None:
            ps = combine_gate_scores(gate_phone_score, ps)
        return apply_behavior_gates(logits, names, ms, ps)

    def forward(self, face, left, right, wheel, mirror, apply_gates=None, class_names=None,
                gate_mirror_score=None, gate_phone_score=None):
        seq, mirror_seq, hand_seq = self._encode_sequences(face, left, right, wheel, mirror)
        frame_logits = self.frame_classifier(seq)
        temporal_logits = self._temporal_logits(seq, frame_logits)

        center = self.center_time_index(seq.size(1))
        mirror_logit = self.mirror_aux(mirror_seq[:, center, :])
        phone_logit = self.phone_aux(hand_seq[:, center, :])

        temporal_logits = self._apply_gates(
            temporal_logits, mirror_logit, phone_logit, apply_gates, class_names,
            gate_mirror_score, gate_phone_score,
        )
        aux = {
            'mirror_logit': mirror_logit,
            'phone_logit': phone_logit,
            'frame_logits': frame_logits,
            'center_index': center,
        }
        return temporal_logits, aux
