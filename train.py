import os
import ray
from ray import air, tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms import ppo
from ray.tune import registry
from environments.al_harvest_env import al_harvest_env_creator
from config import get_experiment_config


# USER INPUT
output_dir = '/home/ben/Downloads/al_harvest_training_output'
num_workers = 4
# the below settings should only be changed if you add support for a new substrate
experiment_name = 'al_harvest'
substrate_name = 'allelopathic_harvest__open'
env_creator = al_harvest_env_creator



os.environ['RAY_memory_monitor_refresh_ms'] = '0'
ray.init(local_mode=False, ignore_reinit_error=True, num_gpus=0)
registry.register_env("meltingpot", env_creator)
default_config = ppo.PPOConfig()
configs, exp_config, tune_config = get_experiment_config(default_config, 
                                                         output_dir, 
                                                         num_workers, 
                                                         experiment_name,
                                                         substrate_name,
                                                         env_creator)

if "WANDB_API_KEY" in os.environ:
    wandb_project = f'{experiment_name}_torch'
    wandb_group = "meltingpot"

    # Set up Weights And Biases logging if API key is set in environment variable.
    wdb_callbacks = [
        WandbLoggerCallback(
            project=wandb_project,
            group=wandb_group,
            api_key=os.environ["WANDB_API_KEY"],
            log_config=True,
        )
    ]
else:
    wdb_callbacks = []
    print("WARNING! No wandb API key found, running without wandb!")

ckpt_config = air.CheckpointConfig(num_to_keep=exp_config['keep'], 
                                   checkpoint_frequency=exp_config['freq'], 
                                   checkpoint_at_end=exp_config['end'], 
                                  )

tuner = tune.Tuner(
        'PPO',
        param_space=configs.to_dict(),
        run_config=air.RunConfig(name = exp_config['name'], callbacks=wdb_callbacks, local_dir=exp_config['dir'], 
                                stop=exp_config['stop'], checkpoint_config=ckpt_config, verbose=0),
    )

results = tuner.fit()

best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
print(best_result)

ray.shutdown()
