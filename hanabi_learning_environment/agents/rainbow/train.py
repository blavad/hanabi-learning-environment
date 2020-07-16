# coding=utf-8
# Copyright 2018 The Dopamine Authors and Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
#
# This file is a fork of the original Dopamine code incorporating changes for
# the multiplayer setting and the Hanabi Learning Environment.
#
"""The entry point for running a Rainbow agent on Hanabi."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import sys
import re

from third_party.dopamine import logger
from hanabi_coop.agent import SimpleAgent, SimpleAgentV2, SimpleAgentV3

import run_experiment
import logging

FLAGS = flags.FLAGS

flags.DEFINE_multi_string(
    'gin_files', [],
    'List of paths to gin configuration files (e.g.'
    '"configs/hanabi_rainbow.gin").')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "DQNAgent.epsilon_train=0.1").')

flags.DEFINE_string('base_dir', None,
                    'Base directory to host all required sub-directories.')

flags.DEFINE_string('checkpoint_dir', '',
                    'Directory where checkpoint files should be saved. If '
                    'empty, no checkpoints will be saved.')
flags.DEFINE_string('checkpoint_file_prefix', 'ckpt',
                    'Prefix to use for the checkpoint files.')
flags.DEFINE_string('logging_dir', '',
                    'Directory where experiment data will be saved. If empty '
                    'no checkpoints will be saved.')
flags.DEFINE_string('logging_file_prefix', 'log',
                    'Prefix to use for the log files.')

flags.DEFINE_string('env', 'Hanabi-Full',
                    'Environment to use.')

flags.DEFINE_string('bot', None,
                    'Bot to use for learning.')

flags.DEFINE_string('intent_ckpt', None,
                    'Checkpoint to intent model.')

flags.DEFINE_string('trust_rate', None,
                    'Trust rate for second part of the training.')


def launch_experiment():
    """Launches the experiment.

    Specifically:
    - Load the gin configs and bindings.
    - Initialize the Logger object.
    - Initialize the environment.
    - Initialize the observation stacker.
    - Initialize the agent.
    - Reload from the latest checkpoint, if available, and initialize the
      Checkpointer object.
    - Run the experiment.
    """
    if FLAGS.base_dir == None:
        raise ValueError('--base_dir is None: please provide a path for '
                         'logs and checkpoints.')

    run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
    experiment_logger = logger.Logger('{}/logs'.format(FLAGS.base_dir))

    environment = run_experiment.create_environment(game_type=FLAGS.env)
    if FLAGS.intent_ckpt is not None:
        environment.load_intent_policy(FLAGS.intent_ckpt)
        environment.trust_rate = FLAGS.trust_rate

    obs_stacker = run_experiment.create_obs_stacker(environment)
    agent = run_experiment.create_agent(environment, obs_stacker, position=1)
    print(FLAGS.bot)
    list_names = re.match(r'\[([\w\W]+)\]', FLAGS.bot)
    if FLAGS.bot is None:
        bot = None
        adv = agent.name
    elif list_names is not None:
        bot_names = re.split(r'\W+', list_names.group(1))
        bot = [getattr(sys.modules[__name__], name)({}, action_form='int')
               for name in bot_names if len(name) > 0]
        adv = bot_names
    else:
        bot = getattr(sys.modules[__name__], FLAGS.bot)({}, action_form='int')
        adv = bot.name

    print("\n\n\n#> Start learning :", agent.name, "vs", adv)
    print("#> Observation size :", obs_stacker.observation_size())
    print("#> Action size :", environment.num_moves())
    print("#> Nb players :", environment.players)
    checkpoint_dir = '{}/checkpoints'.format(FLAGS.base_dir)
    start_iteration, experiment_checkpointer = (
        run_experiment.initialize_checkpointing(agent,
                                                experiment_logger,
                                                checkpoint_dir,
                                                FLAGS.checkpoint_file_prefix))

    run_experiment.run_experiment(agent, bot, environment, start_iteration,
                                  obs_stacker,
                                  experiment_logger, experiment_checkpointer,
                                  checkpoint_dir,
                                  logging_file_prefix=FLAGS.logging_file_prefix)


def main(unused_argv):
    """This main function acts as a wrapper around a gin-configurable experiment.

    Args:
      unused_argv: Arguments (unused).
    """
    launch_experiment()


if __name__ == '__main__':
    app.run(main)
