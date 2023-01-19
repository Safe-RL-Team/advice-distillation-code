from gym.envs.registration import register
from envs.d4rl.d4rl_content.gym_bullet import gym_envs
from envs.d4rl.d4rl_content import infos


for agent in ['hopper', 'halfcheetah', 'ant', 'walker2d']:
    try:
        register(
            id='bullet-%s-v0' % agent,
            entry_point='d4rl_content.gym_bullet.gym_envs:get_%s_env' % agent,
            max_episode_steps=1000,
        )
    except Exception as e:
        if str(e).startswith("Cannot re-register id:"):
            print(f'FYI: Tried to re-register {agent} contents in gym-bullet')
        else:
            raise e

    for dataset in ['random', 'medium', 'expert', 'medium-expert', 'medium-replay']:
        try:
            env_name = 'bullet-%s-%s-v0' % (agent, dataset)
            register(
                id=env_name,
                entry_point='d4rl_content.gym_bullet.gym_envs:get_%s_env' % agent,
                max_episode_steps=1000,
                kwargs={
                    'ref_min_score': infos.REF_MIN_SCORE[env_name],
                    'ref_max_score': infos.REF_MAX_SCORE[env_name],
                    'dataset_url': infos.DATASET_URLS[env_name]
                }
            )
        except Exception as e:
            if str(e).startswith("Cannot re-register id:"):
                print(f'FYI: Tried to re-register {agent} - {dataset} contents in gym-bullet')
            else:
                raise e

