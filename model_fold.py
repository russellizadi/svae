    def forward2(self, x, mask, attr=None):
        """
        Args:
            x (torch.Tensor): (B, L, D) data
            mask (torch.Tensor): (B, L) data: False, padding: True
            attr (torch.Tensor): (B, n_attrs)

        Returns:
            results (dict): {
                'loss': (1),
                'loss_recon': (1),
                'loss_kl': (1),
                'y': (B, L, D),
                'z': (B, n_attrs, z_dim)
            }
        """
        assert x.ndim == 3 and mask.ndim == 2, (x.shape, mask.shape)

        org_mask = mask.clone()
        unmasked = 1. - mask.float()
        patch_L, patch_D = 4, 4

        L, D = x.shape[1], x.shape[2]
        L_pad, D_pad = patch_L - (L % patch_L), patch_D - (D % patch_D)
        L_padded, D_padded = L + L_pad, D + D_pad

        x = F.pad(x, (0, D_pad, 0, L_pad), value=0.)  # B x L_padded x D_padded
        unmasked = F.pad(unmasked, (0, L_pad), value=True)  # B x L_padded

        unfold2d = nn.Unfold(
            kernel_size=(patch_L, patch_D), dilation=1, padding=0,
            stride=(patch_L, patch_D))
        # B x ((L_padded // patch_L) x (D_padded // patch_D)) x (patch_L x patch_D)
        x = unfold2d(x.unsqueeze_(1)).permute(0, 2, 1)

        unfold1d = nn.Unfold(
            kernel_size=(patch_L, 1), dilation=1, padding=0, stride=(patch_L, 1))

        # B x (L_padded // patch_L) x patch_L
        unmasked = unfold1d(unmasked.unsqueeze(
            1).unsqueeze(-1)).permute(0, 2, 1)

        unmasked = torch.prod(unmasked, dim=-1)  # B x (L_padded // patch_L)
        # B x (L_padded // patch_L) x (D_padded // patch_D)
        unmasked = unmasked.repeat_interleave(D_padded // patch_D, dim=-1)

        # B x ((L_padded // patch_L) x (D_padded // patch_D))
        mask = (1. - unmasked).to(bool)

        # encode
        c, durations = self.encoder(x, mask)

        if self.training:
            assert attr is not None
            b = x.shape[0] // 2
            c[:b][attr[:b]] = c[b:][attr[b:]]
            c[b:][attr[b:]] = c[:b][attr[:b]]

        # (?, n_attrs, z_dim), (?, n_attrs)
        z, log_q_z_given_x = self.posterior_q.rsample(c, out_prob=True)
        log_p_z = self.prior.log_prob(z)  # (?, n_attrs)

        param = self.decoder(z, mask, durations)
        # (?, L, d_input), (?, L)
        y, log_p_x_given_z = self.posterior_p.rsample(x, param, out_prob=True)

        kl = log_q_z_given_x - log_p_z  # (?, L)
        kl = kl.sum(1)  # (?)

        unmasked = 1. - mask.float()
        n_unmasked = unmasked.sum(1)  # (?)

        log_p_x_given_z = log_p_x_given_z * unmasked
        log_p_x_given_z = log_p_x_given_z.sum(1)  # (?)

        logp = log_p_x_given_z - kl  # (?)
        logp = logp / n_unmasked  # (?)
        loss = - logp.mean()  # (1)

        loss_recon = - (log_p_x_given_z / n_unmasked).mean()  # (1)
        loss_kl = (kl / n_unmasked).mean()  # (1)

        # fold
        fold2d = nn.Fold(
            output_size=(L_padded, D_padded),
            kernel_size=(patch_L, patch_D), stride=(patch_L, patch_D))
        y = fold2d(y.permute(0, 2, 1)).squeeze_(1)  # B x L_padded x D_padded
        y = y[:, :L, :D]

        org_unmasked = (1. - org_mask.float())
        y = y * org_unmasked.unsqueeze(2)

        results = {
            'loss': loss,
            'loss_recon': loss_recon,
            'loss_kl': loss_kl,
            'y': y,
            'z': z}
        return results