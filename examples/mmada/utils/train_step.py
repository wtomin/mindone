"""Train step wrapper supporting setting drop overflow update, ema etc"""
import logging
from abc import ABCMeta, abstractmethod
from typing import Literal, Optional, Union

import mindspore as ms
from mindspore import nn, ops
from mindspore.amp import DynamicLossScaler, StaticLossScaler
from mindspore.communication.management import GlobalComm
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode

from mindone.diffusers.training_utils import GradAccumulator, GradClipper, GradScaler
from mindone.trainers.ema import EMA
from mindone.trainers.zero import ZeroHelper, prepare_network

logger = logging.getLogger(__name__)


def do_ckpt_combine_online(net_to_save, optimizer_parallel_group=None):
    new_net_to_save = []
    all_gather_op = ops.AllGather(optimizer_parallel_group)
    if optimizer_parallel_group is None:
        logger.warning("Not set zero group, set it WORLD_COMM_GROUP.")
        optimizer_parallel_group = GlobalComm.WORLD_COMM_GROUP
    for item in net_to_save:
        param = item["data"]
        if param.parallel_optimizer:
            new_data = ms.Tensor(all_gather_op(param).asnumpy())
        else:
            new_data = ms.Tensor(param.asnumpy())
        new_net_to_save.append({"name": param.name, "data": new_data})
    return new_net_to_save


def prepare_train_network(
    network: nn.Cell,
    optimizer: nn.Optimizer,
    zero_stage: Literal[0, 1, 2, 3] = 0,
    optimizer_offload: bool = False,
    optimizer_parallel_group: str = None,
    dp_group: str = None,
    comm_fusion: dict = None,
    parallel_modules=None,
):
    """
    Prepare network and optimizer for distributed training.

    Args:
        network (`nn.Cell`): train network, not include grad function,
            grad function must be built after rewrite train network.
        optimizer (`nn.Optimizer`): Must be the subclass of MindSpore Optimizer.
        zero_stage (`int`, *optional*): Stage setting of ZeRO, default is 0.
        optimizer_offload (`bool`, *optional*): Only take effect when optimizer is AdamWeightDecay, default is False.
        optimizer_parallel_group (`str`, *optional*): The name of the optimizer parallel communication group, default is None.
        dp_group (`str`, *optional*): The name of the data parallel communication group, default is None.
        comm_fusion (`dict`, *optional*): A dict contains the types and configurations
            for setting the communication fusion, default is None, turn off the communication fusion. If set a dict,
            turn on the communication fusion.
            Examples: {"allreduce": {"openstate": True, "bucket_size": 5e8},
                       "reduce_scatter": {"openstate": True, "bucket_size": 5e8},
                       "allgather": {"openstate": False, "bucket_size": 5e8},}
        parallel_modules (`dict`, *optional*): A dict of Cells could split parameters in zero3, default is None.
            If None, use `PARALLEL_MODULES` from `mindone.models.modules.parallel`.
    """
    if zero_stage not in [0, 1, 2, 3]:
        raise ValueError("Not support zero_stage {zero_stage}")
    if optimizer_parallel_group is None:
        logger.warning("Not set zero group, set it WORLD_COMM_GROUP.")
        optimizer_parallel_group = GlobalComm.WORLD_COMM_GROUP
    if optimizer_parallel_group != GlobalComm.WORLD_COMM_GROUP and dp_group is None:
        raise ValueError(
            "optimizer_parallel_group {optimizer_parallel_group} and dp_group {dp_group} not full network hccl group coverage"
        )

    is_parallel = _get_parallel_mode() == ParallelMode.DATA_PARALLEL
    if not is_parallel and zero_stage == 0:
        logger.info("No need prepare train_network with zero.")
        zero_helper = None
    else:
        network = prepare_network(network, zero_stage, optimizer_parallel_group, parallel_modules=parallel_modules)
        zero_helper = ZeroHelper(
            optimizer, zero_stage, optimizer_parallel_group, dp_group, optimizer_offload, comm_fusion
        )
    return network, zero_helper


class TrainStep(nn.Cell, metaclass=ABCMeta):
    """
    A base class for training steps in MindSpore with Zero Helper and EMA Helper

    This class provides a basic framework for training neural networks. It takes care of
    gradient accumulation, scaling, and clipping, as well as optimizer updates.

    Args:
        model (nn.Cell): The neural network model to be trained.
        optimizer (nn.Optimizer): The optimizer used for updating model parameters.
        loss_scaler (Optional[Union[StaticLossScaler, DynamicLossScaler]]): The loss scaler to apply during training.
        max_grad_norm (Optional[float]): The maximum gradient norm for gradient clipping.
        gradient_accumulation_steps (Optional[int]): The number of gradient accumulation steps.
        **kwargs: Additional keyword arguments. Available keyword arguments:
            gradient_accumulation_kwargs: Additional keyword arguments for the `GradAccumulator`.
            gradient_checkpointing: whether to apply gradient checkpointing to model

    Attributes:
        model (nn.Cell): The neural network model.
        optimizer (nn.Optimizer): The optimizer.
        parameters (list): The parameters of the optimizer.
        grad_scaler (GradScaler): The gradient scaler.
        grad_clipper (GradClipper): The gradient clipper.
        grad_accumulator (GradAccumulator): The gradient accumulator.

    Properties:
        sync_gradients (bool): Indicates whether gradients are synchronized.

    Methods:
        scale_loss: Scales the loss according to the gradient accumulation steps.
        unscale_loss: Unscales the loss.
        forward: Abstract method for forward pass.
        forward_and_backward: The function for performing forward and backward passes.
        construct: Constructs the training step.

    Raises:
        NotImplementedError: If the forward method is not implemented in subclasses.

    Note:
        - This class is abstract, meaning you must subclass it to create a specific training step.
        - When implementing the forward method, the users must call 'self.scale_loss' at the end.

    Examples:
        1. Basic usage. Only the forward method implementation.

            >>> class MyAwesomeTrainStep(TrainStep):
            >>>     def forward(self, x):
            >>>         y = self.model(x)
            >>>         loss = ops.sum(y)
            >>>         loss = self.scale_loss(loss)
            >>>         return loss, y
            >>>
            >>> model = nn.Dense(10, 10)
            >>> optim = nn.AdamWeightDecay(model.trainable_params())
            >>> train_step = MyAwesomeTrainStep(
            >>>     model,
            >>>     optim,
            >>>     loss_scaler=DynamicLossScaler(2.0**16, 2, 2000),
            >>>     max_grad_norm=1.0,
            >>>     gradient_accumulation_steps=2,
            >>>     gradient_accumulation_kwargs={"length_of_dataloader": 3}
            >>> )
            >>>
            >>> for epoch in range(2):
            >>>     for batch in range(3):
            >>>         inputs = ops.randn(8, 10)
            >>>         outputs = train_step(inputs)

        2. Advanced usage. Multiple model needs to overload __init__ method.

            >>> class MyAwesomeTrainStep(TrainStep):
            >>>     def __init__(self, text_encoder, unet, optim):
            >>>         super().__init__(unet, optim)
            >>>         self.unet = self.model
            >>>         self.text_encoder = text_encoder
            >>>
            >>>     def forward(self, x, t):
            >>>         e = self.text_encoder(t)
            >>>         y = self.unet(x, e)
            >>>         loss = ops.sum(y)
            >>>         loss = self.scale_loss(loss)
            >>>         return loss, y
            >>> # Then you can launch the training as usual.
    """

    def __init__(
        self,
        model: nn.Cell,
        optimizer: nn.Optimizer,
        loss_scaler: Optional[Union[StaticLossScaler, DynamicLossScaler]] = None,
        max_grad_norm: Optional[float] = None,
        gradient_accumulation_steps: Optional[int] = None,
        ema: Optional[EMA] = None,
        zero_helper=None,
        **kwargs,
    ):
        super().__init__()

        self.model = model.set_grad()  # Why do we need call 'set_grad()'?
        if model.jit_config_dict:
            self.set_jit_config(model.jit_config_dict)
        self.optimizer = optimizer
        self.parameters = optimizer.parameters

        self.grad_scaler = GradScaler(loss_scaler)
        self.grad_clipper = GradClipper(max_grad_norm)

        # zero init
        self.zero_helper = zero_helper
        self.zero_stage = zero_helper.zero_stage if zero_helper is not None else 0
        self.run_optimizer = zero_helper.run_optimizer if zero_helper is not None else self.optimizer
        if self.zero_stage != 0:
            self.zero_helper.split_params()

        self.ema = ema
        gradient_accumulation_kwargs = kwargs.pop("gradient_accumulation_kwargs", {})
        self.grad_accumulator = GradAccumulator(
            self.parameters, gradient_accumulation_steps, **gradient_accumulation_kwargs
        )
        # set grad_reducer to Identity if zero_stage > 0
        self.grad_accumulator.grad_reducer = (
            self.grad_accumulator.grad_reducer if self.zero_stage == 0 else nn.Identity()
        )

        self.forward_and_backward = ms.value_and_grad(self.forward, None, weights=self.parameters, has_aux=True)

        gradient_checkpointing = kwargs.get("gradient_checkpointing", False)
        if gradient_checkpointing:
            self.recompute(self.model)
            logger.info("Gradient Checkpointing is applied to model.")

    def recompute(self, b):
        if not b._has_config_recompute:
            b.recompute(parallel_optimizer_comm_recompute=True)
        if isinstance(b, nn.CellList):
            self.recompute(b[-1])
        elif ms.get_context("mode") == ms.GRAPH_MODE:
            b.add_flags(output_no_recompute=True)

    @property
    def sync_gradients(self):
        return self.grad_accumulator.sync_gradients

    def scale_loss(self, loss):
        loss = loss / self.grad_accumulator.gradient_accumulation_steps
        loss = self.grad_scaler.scale(loss)
        return loss

    def unscale_loss(self, loss):
        return self.grad_scaler.unscale(loss)

    @abstractmethod
    def forward(self, *args, **kwargs):
        # You need to scale the loss when performing the model forward pass to create scaled gradients.
        # Do **NOT** forget to include 'loss = self.scale_loss(loss)' after loss calculation!
        ...

    def train_one_step(self, *inputs):
        return self.construct(*inputs)

    def construct(self, *inputs):
        outputs, grads = self.forward_and_backward(*inputs)
        grads = self.grad_accumulator.step(grads)

        if self.sync_gradients:
            # Scaled loss creates scaled gradients. Unscales the gradients.
            grads = self.grad_scaler.unscale(grads)

            # Since the gradients are unscaled, clips as usual.
            grads = self.grad_clipper.clip_grad_norm(grads)

            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            self.grad_scaler.step(self.run_optimizer, grads)

            # Updates the scale for next iteration.
            self.grad_scaler.update()

            # Clear the gradients of accumulator's assigned params.
            self.grad_accumulator.zero_grad()
        if self.ema is not None:
            self.ema.ema_update()
        # The first item of outputs is loss. Unscales the loss for outside logging.
        loss = self.unscale_loss(outputs[0])
        outputs = (loss,) + outputs[1:]
        return outputs


class TrainStepMmaDA(TrainStep):
    def __init__(self, *args, **kwargs):
        self.config = kwargs.pop("config", None)
        assert self.config is not None, "Must pass configugration via key arguments!"

        super().__init__(*args, **kwargs)

    def forward(
        self,
        input_ids: ms.Tensor,
        labels: ms.Tensor,
        batch_size_t2i: int,
        batch_size_lm: int,
        batch_size_mmu: int,
        p_mask_lm: ms.Tensor,
        p_mask_mmu: ms.Tensor,
        answer_lengths: ms.Tensor,
        t2i_masks: ms.Tensor,
    ):
        logits, loss_t2i, loss_lm, loss_mmu = self.model.construct_process(
            input_ids=input_ids,
            labels=labels,
            batch_size_t2i=batch_size_t2i,
            batch_size_lm=batch_size_lm,
            batch_size_mmu=batch_size_mmu,
            max_seq_length=self.config.dataset.preprocessing.max_seq_length,
            p_mask_lm=p_mask_lm,
            p_mask_mmu=p_mask_mmu,
            answer_lengths=answer_lengths,
            t2i_masks=t2i_masks,
        )

        loss = (
            self.config.training.t2i_coeff * loss_t2i
            + self.config.training.lm_coeff * loss_lm
            + self.config.training.mmu_coeff * loss_mmu
        )
        # DO NOT forget to scale loss!
        loss = self.scale_loss(loss)
        return loss, logits, loss_t2i, loss_lm, loss_mmu
