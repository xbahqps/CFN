from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dopamine.discrete_domains import atari_lib
from cpprb import ReplayBuffer, PrioritizedReplayBuffer
#import shannon
import gin
import numpy as np
from torch.utils.tensorboard import SummaryWriter as TorchSummaryWriter
import cv2
import zlib
import time

PSEUDO_COUNT_QUANTIZATION_FACTOR = 8
PSEUDO_COUNT_OBSERVATION_SHAPE = (42, 42)
NATURE_DQN_OBSERVATION_SHAPE = atari_lib.NATURE_DQN_OBSERVATION_SHAPE
TRAIN_START_AMOUNT = 10000

class Timer:
    def __init__(self, name="") -> None:
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        if self.name:
          print(f"{self.name} took {self.interval}")

def timing_wrapper(func):
  def wrap_it(*args, **kwargs):
    with Timer() as t:
      result = func(*args, **kwargs)
    print('{} took {} seconds'.format(func.__name__, t.interval))
    return result
  return wrap_it
"""
def masked_conv2d(inputs, num_outputs, kernel_size,
                  activation_fn=F.relu,
                  weights_initializer=torch.nn.init.xavier_normal_,
                  biases_initializer=torch.zeros,
                  stride=(1, 1),
                  mask_type='A',
                  output_multiplier=1):
    assert mask_type in ('A', 'B') and num_outputs % output_multiplier == 0
    num_inputs = inputs.size(1)
    kernel_shape = (num_outputs, num_inputs) + kernel_size
    strides = (1,) + tuple(stride) + (1,)
    biases_shape = [num_outputs]

    mask_list = [np.zeros(
        (num_outputs // output_multiplier,) + kernel_size + (num_inputs,),
        dtype=np.float32) for _ in range(output_multiplier)]

    for i in range(output_multiplier):
        # Mask type A
        if kernel_size[0] > 1:
            mask_list[i][:, :kernel_size[0] // 2] = 1.0
        if kernel_size[1] > 1:
            mask_list[i][:, kernel_size[0] // 2, :kernel_size[1] // 2] = 1.0
        # Mask type B
        if mask_type == 'B':
            mask_list[i][:, kernel_size[0] // 2, kernel_size[1] // 2] = 1.0

    mask_values = np.concatenate(mask_list, axis=-1)
    mask = torch.tensor(mask_values, dtype=torch.float32)

    w = torch.empty(kernel_shape)
    weights_initializer(w)

    b = torch.empty(biases_shape)
    biases_initializer(b)

    mask = mask.permute(3, 2, 0, 1)  # Adjust dimensions for convolution

    convolution = F.conv2d(inputs, mask * w, stride=stride, padding=(kernel_size[0] // 2, kernel_size[1] // 2))
    convolution_bias = convolution + b.view(1, -1, 1, 1)

    if activation_fn is not None:
        convolution_bias = activation_fn(convolution_bias)

    return convolution_bias
"""
"""
def gating_layer(x, embedding, hidden_units):
    out = masked_conv2d(x, 2 * hidden_units, [3, 3], mask_type='B', activation_fn=None, output_multiplier=2)
    out += nn.Conv2d(embedding.size(1), 2 * hidden_units, kernel_size=1)(embedding)
    out = out.view(-1, 2)
    out = torch.tanh(out[:, 0]) + torch.sigmoid(out[:, 1])
    return out.view(x.size())
"""

# 这个类主要用于存储和管理一个固定大小的数组，其中包含更新次数信息。
# 它提供了方法来添加新的更新次数，增加指定索引处的更新次数，并从指定索引处获取更新次数。
class NumUpdatesBuffer:
  def __init__(self, max_size):
    self.max_size = max_size
    self.num_updates = np.zeros((max_size, ), dtype=np.float32)
    self.ptr = 0
    self.size = 0

  def add(self, num_updates=0.):
    self.num_updates[self.ptr] = num_updates
    self.ptr = (self.ptr + 1) % self.max_size
    self.size = min(self.size + 1, self.max_size)

  def increment_priorities(self, indices):
    assert len(indices.shape) == 1, indices.shape
    self.num_updates[indices] = self.num_updates[indices] + 1

  def sample(self, indices):
    assert len(indices.shape) == 1, indices.shape
    return self.num_updates[indices]
  
# 用于生成二进制数组，模拟硬币翻转的过程。根据初始化时的参数，可以生成具有一定随机性的二进制数组。
class CoinFlipMaker(object):
  def __init__(self, output_dimensions, p_replace, only_zero_flips=False):
    """
        初始化 CoinFlipMaker 类。

        参数：
        - output_dimensions: 输出数组的维度
        - p_replace: 替换元素的概率
        - only_zero_flips: 是否仅生成零翻转的数组，默认为 False
        """
    self.p_replace = p_replace
    self.output_dimensions = output_dimensions
    self.only_zero_flips = only_zero_flips
    self.previous_output = self._draw()

  def _draw(self):
    """
        内部方法：生成新的随机数组。

        返回：
        - 生成的随机数组
        """
    if self.only_zero_flips:
      return np.zeros(self.output_dimensions, dtype=np.float32)
    return 2 * np.random.binomial(1, 0.5, size=self.output_dimensions) - 1

  def __call__(self):
    """
        调用实例，生成新的随机数组。

        返回：
        - 生成的随机数组
        """
    if self.only_zero_flips:
      return np.zeros(self.output_dimensions, dtype=np.float32)
    new_output = self._draw()
    new_output = np.where(
      np.random.rand(self.output_dimensions) < self.p_replace,
      new_output,
      self.previous_output
    )
    self.previous_output = new_output
    return new_output

  def reset(self):
    """
        重置状态，重新生成初始数组。
        """
    if self.only_zero_flips:
      self.previous_output = np.zeros(self.output_dimensions, dtype=np.float32)
    self.previous_output = self._draw()



#############################################################################################################################
#以下是尝试将CFN内在奖励类修改为pytorch版本
#############################################################################################################################
@gin.configurable
class CoinFlipCounterIntrinsicReward_torch(object):
  def __init__(  self,
                 reward_scale,  # 奖励的缩放因子
                 ipd_scale,  # 内在动机的缩放因子
                 observation_shape=NATURE_DQN_OBSERVATION_SHAPE,  # 观察空间的形状，默认为 NATURE_DQN_OBSERVATION_SHAPE
                 resize_shape=PSEUDO_COUNT_OBSERVATION_SHAPE,  # 调整大小后的形状，默认为 PSEUDO_COUNT_OBSERVATION_SHAPE
                 quantization_factor=PSEUDO_COUNT_QUANTIZATION_FACTOR,  # 量化因子，默认为 PSEUDO_COUNT_QUANTIZATION_FACTOR
                 intrinsic_replay_start_size=TRAIN_START_AMOUNT,  # 内在重放缓冲的起始大小，默认为 TRAIN_START_AMOUNT
                 intrinsic_replay_reward_add_start_size=TRAIN_START_AMOUNT,  # 内在重放缓冲的奖励添加起始大小，默认为 TRAIN_START_AMOUNT
                 intrinsic_replay_buffer_size=10**6,  # 内在重放缓冲的大小，默认为 10^6
                 device='cpu',  # 设备类型，默认为 'cpu'
                 output_dimensions=100,  # 输出维度，默认为 100，可能在配置中设置？
                 batch_size=32,  # 批处理大小，默认为 32
                 summary_writer=None,  # 摘要写入器
                 prioritization_strategy="combination",  # 优先级策略，默认为 "combination"
                 use_prioritized_buffer=True,  # 是否使用优先级缓冲，默认为 True
                 priority_alpha=0.5,  # 优先级参数 alpha，默认为 0.5
                 use_final_tanh=False,  # 是否使用最终的 tanh 函数，默认为 False
                 update_period=1,  # 更新周期，默认为 1
                 p_replace=1.,  # 替换概率，默认为 1.0
                 use_random_prior=True,  # 是否使用随机先验，默认为 True
                 use_lwm_representation_learning=False,  # 是否使用 LWM 表示学习，默认为 False
                 lwm_representation_learning_scale=1.0,  # LWM 表示学习的缩放因子，默认为 1.0
                 use_icm_representation_learning=False,  # 是否使用 ICM 表示学习，默认为 False
                 icm_representation_learning_scale=1.0,  # ICM 表示学习的缩放因子，默认为 1.0
                 use_representation_whitening=True,  # 是否使用表示白化，默认为 True
                 use_count_consistency=False,  # 是否使用计数一致性，默认为 False
                 count_consistency_scale=1.0,  # 计数一致性的缩放因子，默认为 1.0
                 use_reward_normalization=False,  # 是否使用奖励归一化，默认为 False
                 shared_representation_learning_latent=False,  # 是否使用共享表示学习的潜在表示，默认为 False
                 bonus_exponent=0.5,  # 奖励指数，默认为 0.5
                 share_dqn_conv=False,  # 是否共享 DQN 卷积，默认为 False
                 agent=None,  # 代理对象
                 use_fresh_rewards=False,  # 是否使用新鲜奖励，默认为 False
                 num_actions=None,  # 动作数量
                 only_zero_flips=False,  # 是否仅零翻转，默认为 False
                 continuous_control=False,  # 是否连续控制，默认为 False
                 use_observation_normalization=False,  # 是否使用观察归一化，默认为 False
                 n_action_dims=None,  # 动作维度
                 ):
       
    # We now have 3 competing prioritization types.
    assert not (continuous_control and use_icm_representation_learning), "can't do cc and icm at this time"
    assert not (continuous_control and share_dqn_conv), "can't do cc and share dqn conv at this time"
    assert prioritization_strategy in ("exponential_average", "equalizing", "combination"), prioritization_strategy
    assert priority_alpha >= 0. and priority_alpha <= 1.
    assert isinstance(n_action_dims, int) or not continuous_control, type(n_action_dims)

    num_repr_enabled = sum(map(int, [use_lwm_representation_learning, use_icm_representation_learning, use_count_consistency]))
    assert num_repr_enabled <= 1, "Only one representation learning method can be enabled at a time."

    if use_fresh_rewards:
      assert not share_dqn_conv, "Can't use fresh rewards and share dqn conv"

    assert intrinsic_replay_reward_add_start_size >= intrinsic_replay_start_size

    self.output_dimensions = output_dimensions
    self.reward_scale = reward_scale
    self.ipd_scale = ipd_scale # we don't actually use this, but don't want to remove yet.
    self.observation_shape = observation_shape
    self.continuous_control = continuous_control

    if isinstance(resize_shape, int):
      resize_shape = (resize_shape, resize_shape)
    self.resize_shape = resize_shape
    self.n_action_dims = n_action_dims
    self.quantization_factor = quantization_factor
    self.optimizer = optim.RMSprop(self.network.parameters(), lr=0.0001, momentum=0.9, eps=0.0001)#pytorch中优化器
    self.intrinsic_replay_start_size = intrinsic_replay_start_size
    self.intrinsic_replay_reward_add_start_size = intrinsic_replay_reward_add_start_size
    self.intrinsic_replay_buffer_size = intrinsic_replay_buffer_size
    self.batch_size = batch_size
    self.use_prioritized_buffer = use_prioritized_buffer
    self.prioritization_strategy = prioritization_strategy
    self.priority_alpha = priority_alpha
    self.use_final_tanh = use_final_tanh
    self.update_period = update_period
    self.summary_writer = summary_writer
    if use_random_prior and use_final_tanh:
      raise Exception("These don't mesh")
    self.use_random_prior = use_random_prior
    self.use_lwm_representation_learning = use_lwm_representation_learning
    self.lwm_representation_learning_scale = lwm_representation_learning_scale
    self.use_representation_whitening = use_representation_whitening
    self.use_count_consistency = use_count_consistency
    self.count_consistency_scale = count_consistency_scale
    self.use_icm_representation_learning = use_icm_representation_learning
    self.icm_representation_learning_scale = icm_representation_learning_scale
    # If true, LWM acts directly on the coinflips!
    self.shared_representation_learning_latent = shared_representation_learning_latent
    self.use_reward_normalization = use_reward_normalization
    self.bonus_exponent = bonus_exponent
    self.share_dqn_conv = share_dqn_conv
    if share_dqn_conv:
      assert tuple(observation_shape) == tuple(resize_shape), "If sharing conv layer, cannot resize"
    self._agent = agent # for sharing the conv... not loving it.
    self.use_fresh_rewards = use_fresh_rewards
    self.num_actions = num_actions
    self.only_zero_flips = only_zero_flips
    self.use_observation_normalization = use_observation_normalization
    self._t = 0

    # when this gets too big, we run out of memory. Shucks.
    # Seems unavoidable, 44*44*8 = 15,000. * 4 (float32) is 60 gb RAM just for that. Jeez. 

    self.previous_obs = None
    self.last_action = None
    self.replay_buffer = self.cc_make_cfn_replay_buffer() if self.continuous_control else self.make_cfn_replay_buffer()
    if self.use_prioritized_buffer:
        self.num_updates_buffer = NumUpdatesBuffer(max_size=self.intrinsic_replay_buffer_size)

    self.coin_flip_maker = CoinFlipMaker(output_dimensions, p_replace, only_zero_flips=only_zero_flips)
    self.channels_dimension = self._agent.stack_size if self.share_dqn_conv else 1
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置设备  
    self.obs_ph = torch.Tensor().to(self.device)
    self.preprocessed_obs = self._preprocess(self.obs_ph, resize_shape)
    self.preprocessed_obs_ph = torch.Tensor().to(self.device)
    self.preprocessed_next_obs_ph = torch.Tensor().to(self.device)
    self.coin_flip_targets = torch.Tensor().to(self.device)
    self.actions_ph = torch.Tensor().to(self.device)
    self.iter_ph = torch.Tensor().to(self.device)
    self.eval_ph = torch.Tensor().to(self.device)
    self.prior_mean = nn.Parameter(torch.zeros(1, self.output_dimensions), requires_grad=False)
    self.prior_var = nn.Parameter(0.002 * torch.ones(1, self.output_dimensions), requires_grad=False)
    self.reward_mean = nn.Parameter(torch.zeros(()), requires_grad=False)
    self.reward_var = nn.Parameter(torch.ones(()), requires_grad=False)
    self.observation_mean = nn.Parameter(torch.zeros(*observation_shape), requires_grad=False)
    self.observation_var = nn.Parameter(torch.ones(*observation_shape), requires_grad=False)
    #self.network = FixedPointCountingNetwork(self._coin_flip_network_template).to(self.device)
    #coin_flip_network = FixedPointCountingNetworkTemplate()  # 根据需要传入参数
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.network = _coin_flip_network_template(device)
    update_dict = self.update()
    self.update_op = update_dict['train_op']
    self.one_over_counts_op = update_dict['one_over_counts']
    self.pred_coin_flip_op = update_dict['flips']
    self.prior_coin_flips_op = update_dict['prior_coin_flips']
    self.loss_op = update_dict['total_loss']
    self.coin_flip_loss_op = update_dict['coin_flip_loss']
    self.lwm_loss_op = update_dict['lwm_loss']
    self.count_consistency_loss_op = update_dict['count_consistency_loss']
    self.icm_loss_op = update_dict['icm_loss']
    self.lwm_output_op = update_dict['lwm_output']  # For logging only
    self.whitening_matrix_op = update_dict['whitening_matrix']
    if self.use_random_prior or self.use_reward_normalization or self.use_observation_normalization:
        self.normalizing_ops = self.make_normalizing_ops()
    self.reward = self.make_reward()
    self._online_summary_ops = self._make_online_summary_ops()
    print(f'Created CFN RewardLearner with n_action_dims={self.n_action_dims}')

  def make_cfn_replay_buffer(self):
    if self.share_dqn_conv:
      buffer_obs_shape = self.resize_shape + (self._agent.stack_size,)
    else:
      buffer_obs_shape = self.resize_shape
    # 定义了存储在回放缓冲区中的数据的格式
    env_dict = dict(
      obs=dict(
        shape = buffer_obs_shape,
        dtype=np.uint8
      ),
      coin_flip=dict(
        shape=(self.output_dimensions,),
        dtype=np.float32
      ),
      act=dict(
        # shape=(1,),
        dtype=np.int32),
    )

    # Create and return the CPPRB buffer
    if self.use_prioritized_buffer:
      return PrioritizedReplayBuffer(
        self.intrinsic_replay_buffer_size,
        env_dict,
        next_of=("obs",),
        stack_compress="obs" if self.share_dqn_conv else None,
        Nstep=False,
        alpha=1.0,
        eps=0.,
      )
    
    return ReplayBuffer(
      self.intrinsic_replay_buffer_size,
      env_dict,
      next_of=("obs",),
      stack_compress="obs" if self.share_dqn_conv else None,
      Nstep=False
    )

  def cc_make_cfn_replay_buffer(self):
    env_dict = dict(
      obs=dict(
        shape=self.observation_shape,
        dtype=np.float32
      ),
      coin_flip=dict(
        shape=(self.output_dimensions,),
        dtype=np.float32,
      ),
      act=dict(
        shape=(self.n_action_dims,),
        dtype=np.float32
      )
    )

    if self.use_prioritized_buffer:
      return PrioritizedReplayBuffer(
        self.intrinsic_replay_buffer_size,
        env_dict,
        next_of=("obs",),
        Nstep=False,
        alpha=1.0,
        eps=0.,
      )

    return ReplayBuffer(
      self.intrinsic_replay_buffer_size,
      env_dict,
      next_of=("obs",),
      Nstep=False
    )

  def _increment_num_updates(self, sample):
    # _store is the data store for memory.

    indices = sample["indexes"]
    self.num_updates_buffer.increment_priorities(indices)

  def _make_online_summary_ops(self):
      network = self.network(self.preproccessed_obs)
      trained_flips = network.trained_coin_flips
      trained_magnitude = torch.sqrt(torch.mean(trained_flips**2, dim=1))
      full_flips = network.coin_flips
      unscaled_bonus = (torch.mean(full_flips**2, dim=1))**self.bonus_exponent
      scaled_bonus = self.reward_scale * (unscaled_bonus - self.reward_mean) / self.reward_var

      if self.use_random_prior:
          full_magnitude = torch.sqrt(torch.mean(full_flips**2, dim=1))
          prior_flips = network.prior_coin_flips
          prior_magnitude = torch.sqrt(torch.mean(prior_flips**2, dim=1))
          rms_prior_mean = torch.sqrt(torch.mean(self.prior_mean**2, dim=1))
          rms_prior_var = torch.sqrt(torch.mean(self.prior_var**2, dim=1))
          return {
            'full_magnitude': full_magnitude,
            'prior_magnitude': prior_magnitude,
            'trained_magnitude': trained_magnitude,
            'rms_prior_mean': rms_prior_mean,
            'rms_prior_var': rms_prior_var,
            'unscaled_reward_mean': self.reward_mean,
            'unscaled_reward_var': self.reward_var,
            'unscaled_bonus' : unscaled_bonus,
            'scaled_bonus': scaled_bonus,
          }
      else:
          return {
            'trained_magnitude': trained_magnitude,
            'unscaled_bonus': unscaled_bonus,
            'scaled_bonus': scaled_bonus,
          }

  def make_normalizing_ops(self):
    assert self.use_random_prior or self.use_reward_normalization, "don't call this without random prior or reward_normalization"
    
    network = self.network(self.preproccessed_obs)
    unscaled_prior_coin_flips = network.unscaled_prior_coin_flips
    unnormalized_unscaled_reward = torch.mean(network.coin_flips**2, dim=1)**self.bonus_exponent
    unnormalized_unscaled_reward = unnormalized_unscaled_reward.squeeze()

    effective_iter = self.iter - self.intrinsic_replay_start_size - 1

    n_ops = 4 if self.use_reward_normalization and self.use_random_prior else 2
    if self.use_observation_normalization:
        n_ops += 2

    def _update():
        moments = []
        if self.use_reward_normalization:
            moments.append((unnormalized_unscaled_reward, self.reward_mean, self.reward_var))
        if self.use_random_prior:
            moments.append((unscaled_prior_coin_flips, self.prior_mean, self.prior_var))
        if self.use_observation_normalization:
            if self.continuous_control:
                preproccessed_obs = self.preproccessed_obs
            else:
                preproccessed_obs = self.preproccessed_obs.float() / 255.
            moments.append((preproccessed_obs, self.observation_mean, self.observation_var))

        ops = []
        for value, mean, var in moments:
            delta = value - mean
            mean.data.add_(delta / effective_iter)
            var_ = var * effective_iter + (delta**2) * effective_iter / (effective_iter + 1)
            var.data = var_ / (effective_iter + 1)
            ops.extend([mean, var])

        return ops

    return n_ops * [torch.tensor(0., dtype=torch.float32)]

  def make_reward(self):
    network = self.network(self.preproccessed_obs)
    flips = network.coin_flips

    mean_square_flips = torch.mean(flips**2, dim=1)
    # count_based bounses
    magnitude = mean_square_flips**self.bonus_exponent

    if self.use_random_prior or self.use_reward_normalization:
        with torch.no_grad():
            self.normalizing_ops()

    reward = magnitude.squeeze()

    if not self.eval_ph:
        return self.reward_scale * (reward - self.reward_mean) / self.reward_var
    else:
        return self.reward_scale * reward


  def make_batch_reward(self, input_tensor):
    resized_input_tensor = self._preprocess(input_tensor, self.resize_shape)
    network = self.network(resized_input_tensor)
    flips = network.coin_flips
    magnitude = torch.mean(flips**2, dim=1)**self.bonus_exponent

    reward = self.reward_scale * magnitude
    reward = reward - self.reward_mean / self.reward_var

    return reward.detach().clone()


  def _add_summary_value(self, name, val, t=None):
    if t is None:
        t = self._t
    self.summary_writer.add_scalar(name, val, t)


  def _convert_obs(self, obs):
    if self.continuous_control:
      assert len(obs.shape) == 1, obs.shape
      return obs[np.newaxis, :]
    elif self.share_dqn_conv:
      # assert obs.shape == (1, 30), obs.shape
      return obs # (1,84, 84, 4) or (1, D)
    else:
      return obs[np.newaxis, :, :, np.newaxis]

  # @timing_wrapper
  def compute_intrinsic_reward(self, observation, training_steps, eval_mode, action=None):
    if self.share_dqn_conv:
        observation = self._agent.state

    if not eval_mode: 
        coin_flip = self.coin_flip_maker()
        padded_observation = self._convert_obs(observation)
        preprocessed_obs = torch.from_numpy(self.preproccessed_obs(padded_observation)).to(self.device)

        if self.previous_obs is not None:
            self.add_to_buffer(self.previous_obs, preprocessed_obs, coin_flip, self.last_action)

        self.previous_obs = preprocessed_obs
        self.last_action = action

        self._t += 1
        if self.replay_buffer.get_stored_size() > self.intrinsic_replay_start_size:
            if self.replay_buffer.get_stored_size() <= self.intrinsic_replay_reward_add_start_size:
                self.update_normalizing_ops(preprocessed_obs, training_steps, eval_mode)
            if self._t % self.update_period == 0:
                self.count_estimator_learning_step()

    if self._t % 50 == 0 and not eval_mode and self.summary_writer is not None:
        logging_observation = self._convert_obs(observation)
        logging_observation = torch.from_numpy(self.preproccessed_obs(logging_observation)).to(self.device)
        online_summary_ops_evaluated = self._online_summary_ops(logging_observation, training_steps, eval_mode)

        self.add_summary_value('CFN/trained_magnitude', online_summary_ops_evaluated['trained_magnitude'])
        self.add_summary_value('CFN/online_unscaled_bonus', online_summary_ops_evaluated['unscaled_bonus'])
        self.add_summary_value('CFN/online_scaled_bonus', online_summary_ops_evaluated['scaled_bonus'])

        if self.use_random_prior:
            self.add_summary_value('CFN/prior_magnitude', online_summary_ops_evaluated['prior_magnitude'])
            self.add_summary_value('CFN/full_magnitude', online_summary_ops_evaluated['full_magnitude'])
            self.add_summary_value('CFN/rms_prior_mean', online_summary_ops_evaluated['rms_prior_mean'])
            self.add_summary_value('CFN/rms_prior_var', online_summary_ops_evaluated['rms_prior_var'])
            self.add_summary_value('CFN/unscaled_reward_mean', online_summary_ops_evaluated['unscaled_reward_mean'])
            self.add_summary_value('CFN/unscaled_reward_var', online_summary_ops_evaluated['unscaled_reward_var'])

    if self.replay_buffer.get_stored_size() > self.intrinsic_replay_reward_add_start_size:
        padded_observation = self._convert_obs(observation)
        padded_observation = torch.from_numpy(self.preproccessed_obs(padded_observation)).to(self.device)
        if self.use_fresh_rewards and not eval_mode:
            if self.use_random_prior or self.use_reward_normalization:
                self.update_normalizing_ops(padded_observation, training_steps, eval_mode)
            return 0.0
        else:
            return self.make_reward(padded_observation, training_steps, eval_mode)
    else:
        return 0.0

  def count_estimator_learning_step(self):
    sample = self.replay_buffer.sample(self.batch_size)

    if self.continuous_control:
        expected_shape = (self.batch_size, self.observation_shape[0])
    elif self.share_dqn_conv:
        expected_shape = (self.batch_size,) + self.resize_shape + (self.channels_dimension,)
    else:
        expected_shape = (self.batch_size,) + self.resize_shape
    assert sample["obs"].shape == expected_shape, sample["obs"].shape

    if self.continuous_control or self.share_dqn_conv:
        preprocessed_obs_batch = torch.from_numpy(sample["obs"]).to(self.device)
    else:
        preprocessed_obs_batch = torch.from_numpy(sample["obs"][..., np.newaxis]).to(self.device)

    coin_flip_batch = torch.from_numpy(sample["coin_flip"]).to(self.device)
    action_batch = torch.from_numpy(sample["act"]).to(self.device)

    if self.continuous_control:
        assert action_batch.shape[0] == self.batch_size, action_batch.shape
        assert len(action_batch.shape) == 2, action_batch.shape
    else:
        assert len(action_batch.shape) == 2, f"action batch shape: {action_batch.shape}"
        action_batch = action_batch[:, 0]

    one_over_counts = self.one_over_counts_op(preprocessed_obs_batch, coin_flip_batch, action_batch)
    mse = self.loss_op(preprocessed_obs_batch, coin_flip_batch, action_batch)
    coin_flip_mse = self.coin_flip_loss_op(preprocessed_obs_batch, coin_flip_batch, action_batch)
    pred_coin_flip = self.pred_coin_flip_op(preprocessed_obs_batch, coin_flip_batch, action_batch)
    update_op = self.update_op(preprocessed_obs_batch, coin_flip_batch, action_batch)
    whitening_matrix = self.whitening_matrix_op(preprocessed_obs_batch, coin_flip_batch, action_batch)

    results = {
        'one_over_counts': one_over_counts,
        'mse': mse,
        'coin_flip_mse': coin_flip_mse,
        'pred_coin_flip': pred_coin_flip,
        'update_op': update_op,
        'whitening_matrix': whitening_matrix
    }

    if self.use_random_prior:
        prior_coin_flips = self.prior_coin_flips_op(preprocessed_obs_batch, coin_flip_batch, action_batch)
        results['prior_coin_flips'] = prior_coin_flips

    if self.use_lwm_representation_learning:
        lwm_mse = self.lwm_loss_op(preprocessed_obs_batch, coin_flip_batch, action_batch)
        lwm_output = self.lwm_output_op(preprocessed_obs_batch, coin_flip_batch, action_batch)
        results['lwm_mse'] = lwm_mse
        results['lwm_output'] = lwm_output

    if self.use_count_consistency:
        preprocessed_next_obs_batch = torch.from_numpy(sample["next_obs"]).to(self.device)
        if not self.continuous_control:
            preprocessed_next_obs_batch = preprocessed_next_obs_batch[..., np.newaxis].to(self.device)
        count_consistency_mse = self.count_consistency_loss_op(preprocessed_obs_batch, preprocessed_next_obs_batch)
        results['count_consistency_mse'] = count_consistency_mse

    if self.use_icm_representation_learning:
        preprocessed_next_obs_batch = torch.from_numpy(sample["next_obs"]).to(self.device)
        if not self.continuous_control:
            preprocessed_next_obs_batch = preprocessed_next_obs_batch[..., np.newaxis].to(self.device)
        icm_loss = self.icm_loss_op(preprocessed_obs_batch, preprocessed_next_obs_batch)
        results['icm_loss'] = icm_loss

    if self.use_prioritized_buffer:
        self.update_priorities(sample, results["one_over_counts"])

    if self.summary_writer is not None:
        if self._t % 50 == 0:
            self.add_summary_value("CFN/mse", results["mse"])
            self.add_summary_value("CFN/coin_flip_mse", results["coin_flip_mse"])
            self.add_summary_value("CFN/buffer_unscaled_bonus", np.mean(results["one_over_counts"] ** self.bonus_exponent))
            self.add_summary_value("CFN/buffer_unscaled_bonus_variance", np.var(results["one_over_counts"] ** self.bonus_exponent))
            self.add_summary_value("CFN/mean_pred_coin_flip", np.mean(results["pred_coin_flip"]))
            self.add_summary_value("CFN/var_pred_coin_flip", np.var(results["pred_coin_flip"]))

            if self.use_random_prior:
                bonus_from_prior = np.mean(np.mean(results["prior_coin_flips"]**2, axis=1) ** self.bonus_exponent)
                self.add_summary_value("CFN/bonus_from_prior", bonus_from_prior)

            if self.use_lwm_representation_learning:
                self.add_summary_value("CFN/lwm_mse", results["lwm_mse"])
                self.add_summary_value("CFN/lwm_output_norm", np.linalg.norm(results["lwm_output"], axis=1).mean())
                lwm_per_dim_mean_mag = np.linalg.norm(results["lwm_output"].mean(axis=0))
                lwm_per_dim_var_mag = np.linalg.norm(results["lwm_output"].var(axis=0))
                self.add_summary_value("CFN/lwm_per_dim_mean_mag", lwm_per_dim_mean_mag)
                self.add_summary_value("CFN/lwm_per_dim_var_mag", lwm_per_dim_var_mag)
                self.add_summary_value("CFN/lwm_whitening_frob_norm", np.linalg.norm(results["whitening_matrix"]).mean())
            if self.use_count_consistency:
                self._add_summary_value("CFN/count_consistency_mse", results["count_consistency_mse"])
            if self.use_icm_representation_learning:
                self._add_summary_value("CFN/icm_loss", results["icm_loss"])

  def update_priorities(self, sample, one_over_counts):
    indices = sample["indexes"]
    num_updates = self.num_updates_buffer[indices].to(self.device)
    assert one_over_counts.shape == num_updates.shape, (one_over_counts.shape, num_updates.shape)

    if self.prioritization_strategy == "combination":
        assert self.priority_alpha >= 0. and self.priority_alpha <= 1.
        # 更新次数和到访次数的加权平均
        new_priorities = (self.priority_alpha / (num_updates + 1)) + (1 - self.priority_alpha) * one_over_counts
    else:
        raise ValueError(f"prioritization strategy {self.prioritization_strategy} not supported")
      
    self.replay_buffer.update_priorities(indices, new_priorities.cpu().numpy())
    if self.prioritization_strategy == "combination":
        self._increment_num_updates(sample)

  def add_to_buffer(self, preprocessed_obs, preprocessed_next_obs, coin_flip, action):
    if self.use_prioritized_buffer:
      # Max priority
      priorities = 1.
      if self.prioritization_strategy == "combination":
        num_updates = 1
        self.replay_buffer.add(
            obs=preprocessed_obs[0],
            next_obs=preprocessed_next_obs[0],
            coin_flip=coin_flip,
            act=action,
            num_updates=num_updates,
            priorities=priorities
        )
      else:
        self.replay_buffer.add(
            obs=preprocessed_obs[0],
            next_obs=preprocessed_next_obs[0],
            coin_flip=coin_flip,
            act=action,
            priorities=priorities
        )
      self.num_updates_buffer.add(num_updates=num_updates)

    else:
      self.replay_buffer.add(
          obs=preprocessed_obs[0],
          next_obs=preprocessed_next_obs[0],
          coin_flip=coin_flip,
          act=action,
        )

  def _preprocess(self, obs, obs_shape):
    """
    Preprocess the input. Normalizes to [0,1]. NOT [0,8] like it was before
    """
    if self.continuous_control:
        assert obs.dtype == torch.float32, obs.dtype
        return obs

    assert obs.dtype == torch.uint8, obs.dtype
    obs = obs.float()
    obs = F.interpolate(obs, size=obs_shape)
    obs = obs.to(torch.uint8)
    return obs


  def _get_conv_info(self, layer_number, size="large"):
    conv_info = {
      'large' : {
        0: {"kernel" : (8, 8), "stride" : 4},
        1: {"kernel" : (4, 4), "stride" : 2},
        2: {"kernel" : (3, 3), "stride" : 1}
      },
      # Medium is probably appropriate for downsampling to 42x42.
      'medium' : {
        0: {"kernel" : (4, 4), "stride" : 2},
        1: {"kernel" : (4, 4), "stride" : 2},
        2: {"kernel" : (3, 3), "stride" : 1}
      },
      # Probably appropriate for 11 x 8
      'small' : {
        0: {"kernel" : (3, 3), "stride" : 1},
        1: {"kernel" : (3, 3), "stride" : 1},
        2: {"kernel" : (3, 3), "stride": 1}
      }
    }

    return conv_info[size][layer_number]

  def _prior_coin_flip_network_function(self, obs, n_conv_layers, kernel_sizes="large"):
    # NOTE: Assume we did the normalization already
    assert obs.dtype == torch.float32, obs.dtype
    assert kernel_sizes in ['small', 'medium', 'large']

    net = obs
    assert n_conv_layers in (0, 1, 2, 3), n_conv_layers
    final_activation = torch.tanh if self.use_final_tanh else None

    # Define orthogonal initialization
    def orthogonal_init(module):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.orthogonal_(module.weight.data)

    # Set layers to not be trainable and apply orthogonal initialization
    for module in self.modules():
        module.apply(orthogonal_init)
        module.requires_grad = False

    if self.continuous_control:
        net = self._cc_prior_network_torso(net)
    else:
        net = self._conv_prior_torso(net, n_conv_layers, kernel_sizes)

    # This doesn't need another layer, eg RND doesn't have one
    prior_coin_flips = nn.Linear(net.size(-1), self.output_dimensions)
    nn.init.orthogonal_(prior_coin_flips.weight.data)

    unscaled_prior_coin_flips = prior_coin_flips(net)
    prior_coin_flips = (prior_coin_flips - self.prior_mean) / (torch.sqrt(self.prior_var + 0.0001))
    prior_coin_flips = prior_coin_flips.detach()
    unscaled_prior_coin_flips = unscaled_prior_coin_flips.detach()
    self.prior_coin_flips = prior_coin_flips
    self.unscaled_prior_coinflips = unscaled_prior_coin_flips

    return prior_coin_flips, unscaled_prior_coin_flips


  def _conv_prior_torso(self, net, n_conv_layers, kernel_sizes):
    if n_conv_layers > 0:
        print("did first conv layer")
        layer_info = self._get_conv_info(0, kernel_sizes)
        kernel, stride = layer_info['kernel'], layer_info['stride']
        net = nn.functional.leaky_relu(self.conv1(net), negative_slope=0.2)

    if n_conv_layers > 1:
        print("did second conv layer")
        layer_info = self._get_conv_info(1, kernel_sizes)
        kernel, stride = layer_info['kernel'], layer_info['stride']
        net = nn.functional.leaky_relu(self.conv2(net), negative_slope=0.2)

    if n_conv_layers > 2:
        print("did third conv layer")
        layer_info = self._get_conv_info(2, kernel_sizes)
        kernel, stride = layer_info['kernel'], layer_info['stride']
        net = nn.functional.leaky_relu(self.conv3(net), negative_slope=0.2)

    net = net.view(net.size(0), -1)  # Flatten the output

    return net

def _cc_prior_network_torso(self, net):
    net = nn.functional.leaky_relu(self.fc1(net), negative_slope=0.2)
    net = nn.functional.leaky_relu(self.fc2(net), negative_slope=0.2)

    return net


@gin.configurable
def _coin_flip_network_template(self, obs, n_conv_layers, kernel_sizes="large", fc_hidden=False, fc_hidden_size=512, stop_conv_grad=False):
    """
    Our network architecture. Will steal mostly from PixelCNN?
    Maybe I won't do that. Maybe I'll do my own architecture

    We get essentially the DQN architecture if we use a hidden FC layer. We get
    the RND architecture if we don't use a hidden layer.
    https://github.com/google/dopamine/blob/master/dopamine/discrete_domains/atari_lib.py
    if input is 84x84x1:
      if n_conv_layers == 0: linear has 84x84=7000 dim
      if n_conv_layers == 1: linear has 7000*32/16=14000 dim
      if n_conv_layers == 2: linear has 14000*2/4=7000 dim
      if n_conv_layers == 3: linear has 7000*1/1=7000 dim
    """
    assert not (self.continuous_control and fc_hidden), "for now no fc_hidden along with continuous control"
    assert kernel_sizes in ['small', 'medium', 'large']
    
    if not self.continuous_control:
        assert obs.dtype == torch.uint8, obs.dtype
        obs = obs.float()  # cast to float
        obs = obs / 255.0  # instead of in preprocess.
        
    if self.use_observation_normalization:
        obs = (obs - self.observation_mean) / (self.observation_var + 1e-6)  # TODO: Not sure if the 1e-6 is necessary
        obs = torch.clamp(obs, min=-5., max=5.)  # Same as RND

    net = obs

    assert n_conv_layers in (0, 1, 2, 3), n_conv_layers
    final_activation = torch.tanh if self.use_final_tanh else None

    # Feature extraction
    if self.continuous_control:
        net = self._cc_network_torso(obs)
    else:
        net = self._conv_network_torso(obs, n_conv_layers, kernel_sizes, stop_conv_grad)

    if fc_hidden:
        net = F.leaky_relu(F.linear(net, fc_hidden_size, weight=torch.zeros_like(net)))

    coin_flips = F.linear(net, self.output_dimensions, weight=torch.zeros_like(net), bias=None)
    trained_coin_flips = coin_flips.clone()

    if self.use_lwm_representation_learning:
        if self.shared_representation_learning_latent:
            # Same output layer!
            lwm_output = coin_flips.clone()
        else:
            lwm_output = F.linear(net, 32, weight=torch.zeros_like(net), bias=None)
    else:
        lwm_output = None

    if self.use_random_prior:
        prior_coin_flips, unscaled_prior_coinflips = self._prior_coin_flip_network_function(obs, n_conv_layers, kernel_sizes=kernel_sizes)
        # TODO: Normalization nonsense
        coin_flips = coin_flips + prior_coin_flips
    else:
        prior_coin_flips, unscaled_prior_coinflips = None, None

    self.coin_flips = coin_flips
    loss = F.mse_loss(self.coin_flip_targets, coin_flips, reduction='mean')

    return coin_flips, loss, trained_coin_flips, prior_coin_flips, unscaled_prior_coinflips, lwm_output, net
   
def cc_network_torso(obs):
    net = nn.functional.leaky_relu(nn.Linear(obs.shape[1], 400))
    net = nn.functional.leaky_relu(nn.Linear(400, 300))
    return net

def conv_network_torso(self, net, n_conv_layers, kernel_sizes, stop_conv_grad):
    if n_conv_layers > 0:
        print("did first conv layer")
        if self.share_dqn_conv:
            net = self._agent.online_convnet.conv1(net)
        else:
            layer_info = self._get_conv_info(0, kernel_sizes)
            kernel, stride = layer_info['kernel'], layer_info['stride']
            net = nn.functional.leaky_relu(nn.Conv2d(net.shape[1], 32, kernel_size=kernel, stride=stride))
        self.conv1_output = net
    if n_conv_layers > 1:
        print("did second conv layer")
        if self.share_dqn_conv:
            net = self._agent.online_convnet.conv2(net)
        else:
            layer_info = self._get_conv_info(1, kernel_sizes)
            kernel, stride = layer_info['kernel'], layer_info['stride']
            net = nn.functional.leaky_relu(nn.Conv2d(net.shape[1], 64, kernel_size=kernel, stride=stride))
        self.conv2_output = net
    if n_conv_layers > 2:
        print("did third conv layer")
        if self.share_dqn_conv:
            net = self._agent.online_convnet.conv3(net)
        else:
            layer_info = self._get_conv_info(2, kernel_sizes)
            kernel, stride = layer_info['kernel'], layer_info['stride']
            net = nn.functional.leaky_relu(nn.Conv2d(net.shape[1], 64, kernel_size=kernel, stride=stride))
        self.conv3_output = net

    net = net.view(net.size(0), -1) # Should have size 2304 when we resize to 42x42.

    if stop_conv_grad:
        net = net.detach()

    return net

def get_whitening_params(self, lwm_out_1, lwm_out_2):
    num_features = self.output_dimensions if self.shared_representation_learning_latent else 32
    identity_matrix = torch.eye(num_features)
    
    def smoothen_matrix(X, I, eps=1e-4):
        return (eps * I)  + ((1-eps) * X)
    
    lwm_concat = torch.cat([lwm_out_1, lwm_out_2], dim=0)
    lwm_mean = torch.mean(lwm_concat, dim=0)[None, ...]  # 添加批次维度
    covariance_matrix = torch.matmul((lwm_concat - lwm_mean).t(), lwm_concat - lwm_mean) / lwm_concat.size(0)
    smooth_cov_matrix = smoothen_matrix(covariance_matrix, identity_matrix)
    L = torch.linalg.cholesky(smooth_cov_matrix)  # A = LL' (L 是下三角矩阵)
    L_inv = torch.linalg.solve(L, identity_matrix)  # 与 W (白化矩阵) 相同
    return lwm_mean, L_inv.transpose(0, 1)  # 这将是 V W.T

def update(self):
    """
    I guess we expect that we'll have our coin_flip placeholder filled here?
    Or, possibly we just fetch a batch and do an update? We can handle the
    `compute_intrinsic_reward` separately I guess.

    I think that's actually better. 
    """
    with torch.no_grad():
        network_output = self.network(self.preprocessed_obs_ph)
        next_network_output = self.network(self.preprocessed_next_obs_ph)
        flips = network_output.coin_flips
        next_flips = next_network_output.coin_flips
        prior_coin_flips = network_output.prior_coin_flips
        one_over_counts = torch.mean(flips**2, dim=1)

        coin_flip_loss = network_output.loss
        lwm_out_1 = network_output.lwm_output
        lwm_out_2 = next_network_output.lwm_output
        if self.use_lwm_representation_learning:
            if self.use_representation_whitening:
                lwm_mean, whitening_matrix = self.get_whitening_params(lwm_out_1=lwm_out_1, lwm_out_2=lwm_out_2)
                lwm_out_1 = torch.matmul((lwm_out_1 - lwm_mean), whitening_matrix)
                lwm_out_2 = torch.matmul((lwm_out_2 - lwm_mean), whitening_matrix)
            lwm_loss = F.mse_loss(lwm_out_1, lwm_out_2)
        else:
            lwm_loss = torch.tensor(0.)
            whitening_matrix = torch.tensor(0.)
        if self.use_count_consistency:
            one_over_next_counts = torch.mean(next_flips**2, dim=1)
            count_consistency_loss = F.mse_loss(one_over_counts, one_over_next_counts)
        else:
            count_consistency_loss = torch.tensor(0.)
        if self.use_icm_representation_learning:
            last_output = network_output.last_layer_output
            next_last_output = next_network_output.last_layer_output
            concatenated_last_outputs = torch.cat([last_output, next_last_output], dim=1)
            concatenated_hidden_layer = F.leaky_relu(self.action_prediction_hidden(concatenated_last_outputs))
            action_prediction_logits = self.action_prediction_logits(concatenated_hidden_layer)
            icm_loss = F.cross_entropy(action_prediction_logits, self.actions_ph)
        else:
            icm_loss = torch.tensor(0.)

    total_loss = coin_flip_loss
    total_loss += (self.lwm_representation_learning_scale * lwm_loss)
    total_loss += (self.count_consistency_scale * count_consistency_loss)
    total_loss += (self.icm_representation_learning_scale * icm_loss)

    train_op = self.optimizer.zero_grad()
    total_loss.backward()
    self.optimizer.step()

    return {
        'train_op': train_op,
        'one_over_counts': one_over_counts,
        'flips': flips,
        'prior_coin_flips': prior_coin_flips,
        'coin_flip_loss': coin_flip_loss,
        'lwm_loss': lwm_loss,
        'total_loss': total_loss,
        'lwm_output': lwm_out_1,
        'whitening_matrix': whitening_matrix,
        'count_consistency_loss': count_consistency_loss,
        'icm_loss': icm_loss,
    }
