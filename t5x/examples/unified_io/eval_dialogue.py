import argparse
import os
from multiprocessing.pool import ThreadPool
from typing import List

from fabric import Connection

from settings import TPU_HOSTS, TFDS_DIR
from t5x.examples.unified_io.data.task_constants import FINETUNE_OUTPUT_FEATURES
from t5x.examples.unified_io.launch_finetuning_tasks import get_connections


def get_eval_command(
    task, checkpoint_path, n_nodes, split, host_name,
    model_size=None,  # Defaults depends on the checkpoint name
    do_sampling=None, batch_size=None,
    gin_args=None, num_examples=None,
):
  """Build a python command to evaluate on a task. We have so many tasks having a config file
  for each one seems a bit insane, so we use some more general purpose config files and
  use command line args to add task-specific modification.

  Tries to pick sensible default based on the tasks for things like batch size and sampling,
  but defaults might not be correct for all tasks.
  """
  num_partitions = 1
  if model_size is None:
    if "baseline" in checkpoint_path:
      if "/xl" in checkpoint_path:
        model_size = "xl"
        num_partitions = 4
      elif "/base" in checkpoint_path:
        model_size = "base"
      elif "/small" in checkpoint_path:
        model_size = "small"
      elif "/large" in checkpoint_path:
        model_size = "large"
      else:
        raise NotImplementedError
    else:
      if "/xl" in checkpoint_path:
        model_size = "multi_xl"
        num_partitions = 4
      elif "/base" in checkpoint_path:
        model_size = "multi_base"
      elif "/small" in checkpoint_path:
        model_size = "multi_small"
      elif "/large" in checkpoint_path:
        model_size = "multi_large"
      else:
        raise NotImplementedError
    print(f"Defaulting model size to {model_size} based on checkpoint name")

  model_config = f"t5x/examples/unified_io/t5_1_1/{model_size}.gin"

  assert checkpoint_path.count("checkpoint_") == 1
  eval_output_dir = checkpoint_path.rstrip("/").replace("checkpoint_", "checkpoint-") + f"-{task}-eval"

  if gin_args is None:
    gin_args = {}

  gin_args.update({
    "MIXTURE_OR_TASK_NAME": task,
    "CHECKPOINT_PATH": checkpoint_path,
    "EVAL_OUTPUT_DIR": eval_output_dir,
    "utils.DatasetConfig.split": split,
    "partitioning.PjitPartitioner.num_partitions": num_partitions,
  })

  eval_config = "eval_dialogue.gin"
  eval_config = os.path.join("t5x/examples/unified_io/t5_1_1/eval", eval_config)

  example_per_core = 1

  if num_examples is not None:
    gin_args["evaluate.num_examples"] = num_examples

  if num_partitions is not None:
    gin_args["partitioning.PjitPartitioner.num_partitions"] = num_partitions

  if batch_size is not None:
    gin_args["utils.DatasetConfig.batch_size"] = batch_size
  else:
    print(f"Defaulting batch to {example_per_core * n_nodes * 8}")
    gin_args["utils.DatasetConfig.batch_size"] = example_per_core * n_nodes * 8

  command = f"python3 /home/{host_name}/t5x/eval.py --tfds_data_dir={TFDS_DIR} --gin_file={model_config} --gin_file={eval_config}"
  for arg_name, value in gin_args.items():
    # Hack to catch gin references that use "@"
    if isinstance(value, str) and "@" not in value:
      value = "\\\"" + value + "\\\""
    command += f" --gin.{arg_name}={value}"
  return command


def run_on_connections(conns: List[Connection], command, disown=True):
  """Given set of connections to a TPU, run a command or commands on the TPU"""

  if isinstance(command, str):
    print(f"Will launch command: {command} on {len(conns)} connecton")
  else:
    print(f"Launched {len(command)} commands")
    for name, sub_command in command:
      print(f"{name}: {sub_command}")

  def _run(conn, _command, _disown=True, out_name=None):
    if out_name is None:
      out_name = "out.txt"
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    with conn.cd('~'):
      # Redirect stderr to stdout
      to_run = f'{_command} 2>&1'
      # Filter for annoying and impossible-to-turn-off google auth messages
      to_run +=  '| grep --line-buffered -v "$tensorstore/internal/oauth2/google_auth_provider.cc"'
      # pipe to a log file
      if disown:
        to_run += f' > {out_name}'
      else:
        to_run += f' | tee {out_name}'  # So stdout also gets piped back through `conn`

      if wandb_api_key is not None:
        # Set the api key first
        to_run = f"export WANDB_API_KEY={wandb_api_key}; " + to_run

      print(conn.host, "Running: " + to_run)
      result = conn.run(to_run, disown=_disown)
      if not _disown and result.exited != 0:
        raise ValueError(f"Command exited with code {result}")

  if isinstance(command, str):
    if len(conns) == 1:
      _run(conns[0], command, disown)
    else:
      if disown:
        with ThreadPool(processes=len(conns)) as p:
          p.starmap(_run, [(c, command) for c in conns])
      else:
        # Run the first connect without disown
        with ThreadPool(processes=len(conns)-1) as p:
          future = p.starmap_async(_run, [(c, command) for c in conns[1:]])
          _run(conns[0], command, disown)
          future.get()
  else:
    # Cannot disown since we need to know when the previous command
    # ended before starting the next one
    assert not disown
    for ix, (name, sub_command) in enumerate(command):
      print()
      print("*"*40)
      print(f"Starting command {ix+1}/{len(command)}")
      output_file = f"{name}-out.txt"
      if len(conns) == 1:
        _run(conns[0], sub_command, disown, output_file)
      else:
        # Only run the first connect without disown
        with ThreadPool(processes=len(conns)-1) as p:
          future = p.starmap_async(_run, [(c, sub_command) for c in conns[1:]])
          _run(conns[0], sub_command, False)
          future.get()

  if disown:
    print(f"Done, model should be running on {[x.host for x in conns]}")


def run_eval():
  parser = argparse.ArgumentParser()
  parser.add_argument("host", choices=TPU_HOSTS)
  parser.add_argument("--split", default="test")
  parser.add_argument("--checkpoint", "-c", type=str, required=True)
  parser.add_argument("--task", default="demo_nlp")
  parser.add_argument("--num_examples", default=None, type=int)
  parser.add_argument("--host_name", required=True)
  args = parser.parse_args()

  assert FINETUNE_OUTPUT_FEATURES["image_inputs"].rank == 2

  host = args.host

  zone = "us-east1-d"
  conns = get_connections(host, zone)

  command = get_eval_command(
    args.task,
    args.checkpoint,
    len(conns),
    split=args.split,
    host_name=args.host_name,
    do_sampling=False,
    num_examples=args.num_examples,
  )

  run_on_connections(
    conns,
    command,
    disown=False,
  )


if __name__ == '__main__':
  run_eval()
