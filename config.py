from ray.rllib.policy import policy
from meltingpot import substrate

def get_experiment_config(default_config, output_dir, num_workers, experiment_name,
                          substrate_name, env_creator):
    scale_factor = 8
    player_roles = substrate.get_config(substrate_name).default_player_roles 
    run_configs = default_config
    experiment_configs = {}
    tune_configs = None
    run_configs.num_rollout_workers = num_workers
    run_configs.num_gpus = 0

    # Training
    run_configs = run_configs.framework('torch')
    run_configs.log_level = 'INFO'
    run_configs._disable_preprocessor_api = True
    run_configs.model["use_lstm"] = True

    # Environment
    run_configs.env = "meltingpot"
    run_configs.env_config = {"substrate": substrate_name, "roles": player_roles, "scaled": scale_factor}

    base_env = env_creator(run_configs.env_config)
    policies = {
        "default_policy": policy.PolicySpec(
            observation_space=base_env.observation_space["player_0"],
            action_space=base_env.action_space["player_0"],
            config={
                "model": {
                    "conv_filters": [
                        [16, [3, 3], 1],
                        [32, [3, 3], 1]
                    ],
                },
            }
            )
    }

    # Define a policy mapping function to map all agents to 'default_policy'
    def policy_mapping_fn(agent_id, *args, **kwargs):
        return "default_policy"

    run_configs.multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)

    # Experiment Trials
    experiment_configs['name'] = experiment_name
    experiment_configs['stop'] = {}
    experiment_configs['keep'] = 500
    experiment_configs['freq'] = 2
    experiment_configs['end'] = False
    experiment_configs['dir'] = f"{output_dir}/torch"
 
    return run_configs, experiment_configs, tune_configs
