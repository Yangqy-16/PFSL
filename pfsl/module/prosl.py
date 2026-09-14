from .cls_module import *
from tqdm import tqdm


# @MODULE_REGISTRY.register()
# class ProSLModule(ClsModule):
#     def forward(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
#         g1_idx, g1, g1_mask, g1_func = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda()
#         g2_idx, g2, g2_mask, g2_func = batch[4].cuda(), batch[5].cuda(), batch[6].cuda(), batch[7].cuda()
#         label = batch[8].cuda()
#         cell_embedding = batch[9].cuda() if self.cfg.DATASET.CELL_LINE else None

#         logit, attns = self.model(g1, g1_mask, g1_func, g2, g2_mask, g2_func, cell_embedding)

#         loss = self.loss_func(logit, label)
#         return loss, logit, label, g1_idx, g2_idx


# @MODULE_REGISTRY.register()
# class PFModule(ClsModule):
#     def forward(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
#         g1_idx, g1_func = batch[0].cuda(), batch[1].cuda()
#         g2_idx, g2_func = batch[2].cuda(), batch[3].cuda()
#         label = batch[4].cuda()

#         logit = self.model(g1_func, g2_func)

#         loss = self.loss_func(logit, label)
#         return loss, logit, label, g1_idx, g2_idx


@MODULE_REGISTRY.register()
class PFnewModule(ClsModule):
    # @configurable
    # def __init__(self, model: nn.Module, cfg: DictConfig):
    #     super().__init__(model, cfg)

        # self.att_a = defaultdict(list)
        # self.att_b = defaultdict(list)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        # g1_func, g1_mask = batch[0].cuda(), batch[1].cuda()
        # g2_func, g2_mask = batch[2].cuda(), batch[3].cuda()
        # label = batch[4].cuda()
        # g1_idx, g2_idx = batch[5].cuda(), batch[6].cuda()
        for key, value in batch.items():
            if value is not None:
                # print(key, value.shape)
                batch[key] = value.cuda()

        logit, atts = self.model(batch)#, None #g1_func, g2_func, g1_mask, g2_mask

        loss = self.loss_func(logit, batch['label'])
        return loss, logit, batch['label'], batch['g1_idx'], batch['g2_idx'], atts

    def step_common(self, batch: dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor], stage: RunningStage, stage_str: str) -> dict[str, Any]:
        loss, logit, label, g1_idx, g2_idx, atts = self.forward(batch)

        if stage != RunningStage.TESTING:
            self.log(f"{stage_str}/loss", loss, prog_bar=True, on_step=True, on_epoch=False, batch_size=len(label), sync_dist=True, rank_zero_only=True)

        self.losses[stage].append((loss.detach().item(), len(label)))
        self.logits[stage].append(logit.detach().cpu())
        self.labels[stage].append(label.detach().cpu())
        self.g1_indices[stage].append(g1_idx.detach().cpu())
        self.g2_indices[stage].append(g2_idx.detach().cpu())
        # if stage == RunningStage.TESTING:
        #     att_a, att_b = atts
        #     self.att_a[stage].append(att_a.detach().cpu())
        #     self.att_b[stage].append(att_b.detach().cpu())

        return {"loss": loss}

    # def end_common(self, stage: RunningStage, stage_str: str) -> None:
    #     ###### Original codes: START ######
    #     ml = mean_loss(self.losses[stage])
    #     labels = torch.cat(self.labels[stage])  # [N]
    #     logits = torch.cat(self.logits[stage])  # [N]
        
    #     probs = torch.sigmoid(logits)
    #     metrics = compute_metrics(labels, probs)

    #     self.log(f"{stage_str}/mean_loss", ml, on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
    #     self.log(f"{stage_str}/auroc", metrics['auroc'], on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
    #     self.log(f"{stage_str}/auprc", metrics['auprc'], on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
    #     if stage == RunningStage.VALIDATING:
    #         if self.monitor == 'avgmtr':
    #             self.log(self.monitor, (metrics['auroc'] + metrics['auprc']) / 2, logger=False, on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
    #         elif self.monitor != 'loss':
    #             self.log(self.monitor, metrics[self.monitor.lower()], logger=False, on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
    #         else:
    #             self.log(self.monitor, ml, logger=False, on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)
    #     if stage == RunningStage.TESTING:
    #         for metric in ['acc', 'precision', 'recall', 'f1', 'bacc']:
    #             self.log(f"{stage_str}/{metric}", metrics[metric], on_step=False, on_epoch=True, sync_dist=True, rank_zero_only=True)

    #         g1_indices = torch.cat(self.g1_indices[stage])
    #         g2_indices = torch.cat(self.g2_indices[stage])
    #         df = pd.DataFrame({'gene1': g1_indices.detach().cpu().numpy(), 'gene2': g2_indices.detach().cpu().numpy(), 'probs': probs.detach().cpu().numpy()})
    #         output_dir = f'{self.cfg.OUTPUT_DIR}/csv_log'
    #         version_num = len(os.listdir(output_dir)) - 1
    #         df.to_csv(f'{output_dir}/version_{version_num}/test_logits.csv', index=False)
    #         # np.save(f'{output_dir}/version_{version_num}/test_metrics.npy', metrics, allow_pickle=True)

    #     self.losses[stage].clear()
    #     self.logits[stage].clear()
    #     self.labels[stage].clear()
    #     # NOTE: keep indices!
    #     ###### Original codes: END ######

    #     # att_save_dir = f'{self.cfg.OUTPUT_DIR}/att'
    #     # os.makedirs(att_save_dir, exist_ok=True)

    #     # if stage == RunningStage.TESTING:
    #     #     g1_indices = torch.cat(self.g1_indices[stage])
    #     #     g2_indices = torch.cat(self.g2_indices[stage])

    #     #     self.att_a[stage] = torch.cat(self.att_a[stage])
    #     #     self.att_b[stage] = torch.cat(self.att_b[stage])

    #     #     for g1, att_a, g2, att_b in tqdm(zip(g1_indices, self.att_a[stage], g2_indices, self.att_b[stage]), total=len(g1_indices)):
    #     #         if ((g1 == 672 or g1 == 675) and g2 == 142) \
    #     #             or (g1 == 3309 and g2 == 3845) \
    #     #             or (g1 == 5685 and g2 == 6598):
    #     #             torch.save(att_a, f'{att_save_dir}/{g1}_{g2}: {g1}.pt')
    #     #             torch.save(att_b, f'{att_save_dir}/{g1}_{g2}: {g2}.pt')
