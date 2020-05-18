import tensorflow as tf
import numpy as np
import time
from meta_mb.logger import logger


class Trainer(object):
    """
    Performs steps for MAML

    Args:
        algo (Algo) :
        env (Env) :
        sampler (Sampler) : 
        sample_processor (SampleProcessor) : 
        baseline (Baseline) : 
        policy (Policy) : 
        n_itr (int) : Number of iterations to train for
        start_itr (int) : Number of iterations policy has already trained for, if reloading
        num_inner_grad_steps (int) : Number of inner steps per maml iteration
        sess (tf.Session) : current tf session (if we loaded policy, for example)
    """
    def __init__(
            self,
            algo,
            env,
            sampler,
            sample_processor,
            policy,
            n_itr,
            start_itr=0,
            task=None,
            sess=None,
            use_rp_inner=False,
            use_rp_outer=False,
            reward_threshold=0.8,
            ):
        self.algo = algo
        self.env = env
        self.sampler = sampler
        self.sample_processor = sample_processor
        self.baseline = sample_processor.baseline
        self.policy = policy
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.task = task
        if sess is None:
            sess = tf.Session()
        self.sess = sess
        self.use_rp_inner = use_rp_inner
        self.use_rp_outer = use_rp_outer
        self.reward_threshold = reward_threshold
        self.curriculum_step = 0

    def check_advance_curriculum(self, data):
        rewards = data['avg_reward']
        should_advance = rewards > self.reward_threshold
        if should_advance:
            self.curriculum_step += 1
        return self.curriculum_step

    def train(self):
        """
        Trains policy on env using algo

        Pseudocode:
            for itr in n_itr:
                for step in num_inner_grad_steps:
                    sampler.sample()
                    algo.compute_updated_dists()
                algo.optimize_policy()
                sampler.update_goals()
        """
        with self.sess.as_default() as sess:

            # initialize uninitialized vars  (only initialize vars that were not loaded)
            uninit_vars = [var for var in tf.global_variables() if not sess.run(tf.is_variable_initialized(var))]
            sess.run(tf.variables_initializer(uninit_vars))
            advance_curriculum = False

            start_time = time.time()
            for itr in range(self.start_itr, self.n_itr):
                self.sampler.update_tasks()
                itr_start_time = time.time()
                logger.log("\n ---------------- Iteration %d ----------------" % itr)
                logger.log("Sampling set of tasks/goals for this meta-batch...")

                """ -------------------- Sampling --------------------------"""

                logger.log("Obtaining samples...")
                time_env_sampling_start = time.time()
                paths = self.sampler.obtain_samples(log=True, log_prefix='train-', advance_curriculum=advance_curriculum)
                sampling_time = time.time() - time_env_sampling_start

                """ ----------------- Processing Samples ---------------------"""

                logger.log("Processing samples...")
                time_proc_samples_start = time.time()
                samples_data = self.sample_processor.process_samples(paths, log='all', log_prefix='train-')
                advance_curriculum = self.check_advance_curriculum(samples_data)
                proc_samples_time = time.time() - time_proc_samples_start

                """ ------------------ Reward Predictor Splicing ---------------------"""
                r_discrete, logprobs = self.algo.reward_predictor.get_actions(samples_data['env_infos']['next_obs_rewardfree'])
                # Splice into the inference process
                if self.use_rp_inner:
                    samples_data['observations'][:,:, -2] = r_discrete[:, :, 0]
                # Splice into the meta-learning process
                if self.use_rp_outer:
                    samples_data['rewards'] = logprobs[:, :, 1]   
                
                """ ------------------ End Reward Predictor Splicing ---------------------"""

                if type(paths) is list:
                    self.log_diagnostics(paths, prefix='train-')
                else:
                    self.log_diagnostics(sum(paths.values(), []), prefix='train-')

                """ ------------------ Policy Update ---------------------"""

                logger.log("Optimizing policy...")
                # This needs to take all samples_data so that it can construct graph for meta-optimization.
                time_optimization_step_start = time.time()
                self.algo.optimize_policy(samples_data)
                # TODO: Make sure we optimize this for more steps
                self.algo.optimize_reward(samples_data)

                """ ------------------- Logging Stuff --------------------------"""
                logger.logkv('Itr', itr)
                logger.logkv('n_timesteps', self.sampler.total_timesteps_sampled)

                logger.logkv('Time-Optimization', time.time() - time_optimization_step_start)
                logger.logkv('Time-SampleProc', np.sum(proc_samples_time))
                logger.logkv('Time-Sampling', sampling_time)

                logger.logkv('Time', time.time() - start_time)
                logger.logkv('ItrTime', time.time() - itr_start_time)

                logger.logkv('Curriculum Step', self.curriculum_step)
                logger.logkv('Curriculum Percent', self.curriculum_step / len(self.env.levels_list))

                logger.log("Saving snapshot...")
                params = self.get_itr_snapshot(itr)
                logger.save_itr_params(itr, params)
                logger.log("Saved")

                logger.dumpkvs()
                if itr == 0:
                    sess.graph.finalize()

        logger.log("Training finished")
        self.sess.close()

    def get_itr_snapshot(self, itr):
        """
        Gets the current policy and env for storage
        """
        if self.algo.reward_predictor is not None:
            return dict(itr=itr, policy=self.policy, env=self.env, baseline=self.baseline)
        else:
            return dict(itr=itr, policy=self.policy, env=self.env, baseline=self.baseline, reward_predictor=self.algo.reward_predictor)

    def log_diagnostics(self, paths, prefix):
        # TODO: we aren't using it so far
        self.env.log_diagnostics(paths, prefix)
        self.policy.log_diagnostics(paths, prefix)
        self.baseline.log_diagnostics(paths, prefix)
