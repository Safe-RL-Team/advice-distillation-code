from algos.agent import Agent
from algos.ppo import PPOAgent
from algos.sac import SACAgent
from algos.mf_trainer import Trainer
from scripts.arguments import *
from envs.babyai.utils.obs_preprocessor import make_obs_preprocessor
from scripts.test_generalization import make_log_fn
from algos.data_collector import DataCollector
from utils.rollout import rollout

import shutil
from logger import logger
from utils.utils import set_seed
from envs.babyai.levels.iclr19_levels import *
from envs.babyai.levels.envdist import EnvDist
import pathlib
import joblib
import os

from train_model import *
from utils.agent_loader import *


def load_saved_agent(args):
    return args, load_agent_iteration(args.saved_iteration)


def run_distillation(args):
    args, saved_agent = load_saved_agent(args)
    original_args = saved_agent.args

    if not hasattr(args, 'noise'):
        args.noise = False
    exp_name = args.prefix
    set_seed(args.seed)
    feedback_list = get_feedback_list(args)
    env = make_env(args, feedback_list)
    args.feedback_list = feedback_list
    obs_preprocessor = make_obs_preprocessor(feedback_list)

    # Either we need an existing dataset, or we need to collect
    assert (args.buffer_path or (args.collect_policy is not None) or
            (args.rl_teacher is not None and args.collect_with_rl_policy) or
            (args.distill_teacher is not None and args.collect_with_distill_policy))
    # We can't collect with both policies
    assert not (args.collect_with_rl_policy and args.collect_with_distill_policy)


    log_policy = None
    if args.rl_teacher is not None:
        rl_agent = create_agent(args.rl_policy, args.rl_teacher, env, args,
                                 obs_preprocessor)
        log_policy = rl_agent
    else:
        rl_agent = None
    if args.distill_teacher is not None:
        distilling_agent = create_agent(args.distill_policy, args.distill_teacher, env, args, obs_preprocessor)
        log_policy = distilling_agent
    else:
        distilling_agent = None
    if args.relabel_teacher is not None:
        relabel_policy = create_agent(args.relabel_policy, args.relabel_teacher, env, args, obs_preprocessor)
    else:
        relabel_policy = None

    if args.collect_with_rl_policy:
        collect_policy = rl_agent
        args.collect_teacher = args.rl_teacher
    elif args.collect_with_distill_policy:
        collect_policy = distilling_agent
        args.collect_teacher = args.distill_teacher
    elif args.collect_teacher is not None:
        collect_policy = create_agent(args.collect_policy, args.collect_teacher, env, args, obs_preprocessor)
        if log_policy is None:
            log_policy = collect_policy
    else:
        collect_policy = None

    exp_dir = os.getcwd() + '/logs/' + exp_name
    args.exp_dir = exp_dir
    is_debug = args.prefix == 'DEBUG'
    configure_logger(args, exp_dir, args.start_itr, is_debug)

    if args.eval_envs is not None:
        eval_policy(log_policy, env, args, exp_dir)
        return

    envs = [env.copy() for _ in range(args.num_envs)]
    for i, new_env in enumerate(envs):
        new_env.seed(i+100)
        new_env.set_task()
        new_env.reset()
    if collect_policy is None:
        sampler = None
    else:
        sampler = DataCollector(collect_policy, envs, args)

    buffer_name = exp_dir if args.buffer_path is None else args.buffer_path
    args.buffer_name = buffer_name
    num_rollouts = 1 if is_debug else args.num_rollouts
    log_fn = make_log_fn(env, args, 0, exp_dir, log_policy, hide_instrs=args.hide_instrs, seed=args.seed+1000,
                         stochastic=True, num_rollouts=num_rollouts, policy_name=exp_name,
                         env_name=str(args.level),
                         log_every=args.log_interval)

    trainer = Trainer(
        args=args,
        collect_policy=collect_policy,
        rl_policy=rl_agent,
        distill_policy=distilling_agent,
        relabel_policy=relabel_policy,
        sampler=sampler,
        env=deepcopy(env),
        obs_preprocessor=obs_preprocessor,
        log_dict=log_dict,
        log_fn=log_fn,
    )
    trainer.train()


if __name__ == '__main__':
    parser = DistillArgumentParser()
    args = parser.parse_args()
    run_distillation(args)

