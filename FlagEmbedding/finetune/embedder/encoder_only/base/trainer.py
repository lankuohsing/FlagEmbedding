import os
import torch
import logging
from typing import Optional
from torch import nn
from typing import Union, Any
from FlagEmbedding.abc.finetune.embedder import AbsEmbedderTrainer

logger = logging.getLogger(__name__)


class EncoderOnlyEmbedderTrainer(AbsEmbedderTrainer):
    """
    Trainer class for base encoder models.
    """
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """Save the model to directory.

        Args:
            output_dir (Optional[str], optional): Output directory to save the model. Defaults to ``None``.

        Raises:
            NotImplementedError
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, 'save'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} '
                f'does not support save interface')
        else:
            self.model.save(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        # save the checkpoint for sentence-transformers library
        # if self.is_world_process_zero():
        #     save_ckpt_for_sentence_transformers(output_dir,
        #                                         pooling_mode=self.args.sentence_pooling_method,
        #                                         normlized=self.args.normlized)

    def prediction_step(
            self,
            model: nn.Module,
            inputs: dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[list[str]] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        '''
        在transformers的traine里面会这么调用：
        losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
        '''
        # 强制启用return_loss参数
        inputs["return_loss"] = True  # 关键修改点 ▼

        # 原始逻辑（保持其他判断不变）
        has_labels = any(inputs.get(k) is not None for k in self.label_names)
        return_loss = inputs.get("return_loss", None)

        # 执行原始prediction_step逻辑
        loss, logits, labels = super().prediction_step(
            model,
            inputs,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys)


        # 新增：如果loss为None，尝试从outputs提取
        if loss is None:
            with torch.no_grad():
                outputs = model(**inputs)
                if isinstance(outputs, dict) and "loss" in outputs:
                    loss = outputs["loss"].mean().detach()

        return (loss, logits, labels)# 只有loss有有效值
