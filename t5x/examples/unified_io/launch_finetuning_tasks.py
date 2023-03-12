import json
import os
import subprocess
from multiprocessing.pool import ThreadPool
from os.path import join, expanduser, dirname, relpath
from typing import List

from fabric import Connection


def setup_tpu(conn: Connection, tpu_home, transfer_files="conn", install_deps=True):
  """Initialize a TPU so it is ready to run T5X code"""

  # Delete any running processes
  if conn.run('pkill train.py', warn=True, hide=True).exited == 0:
    print(conn.host, "Running instance of train.py was found and killed",)

  if conn.run('pkill eval.py', warn=True, hide=True).exited == 0:
    print(conn.host, "Running instance of eval.py was found and killed",)

  # Delete existing code base if it exists
  remote_code_path = f"/home/{tpu_home}/t5x"
  if conn.run(f'rm -rf {remote_code_path}/*', warn=True, hide=True).exited == 0:
    print(conn.host, "t5x directory found and was removed")

  # Find the root of the local t5x directory
  local_code_path = __file__
  local_code_path = os.path.abspath(local_code_path)
  for _ in range(4):
    local_code_path = dirname(local_code_path)

  if transfer_files == "conn":
    # This is slower then rsync for me, but has the potential advantage of using `conn` instead
    # running a seperate process so it should be or reliable
    print(conn.host, "Copy files over individually...")
    conn.run(f'mkdir {remote_code_path}')
    for dirpath, dirnames, filenames in os.walk(local_code_path):
      # Modify dirname in-place to remove directories we want to skip copying over
      # We only copy metadata, t5x, t5x/configs, t5x/examples
      if dirpath == local_code_path:
        del dirnames[2:]
        dirnames[:] = ["metadata", "t5x"]
      if dirpath == join(local_code_path, "t5x"):
        del dirnames[2:]
        dirnames[:] = ["configs", "examples"]

      # Copy over any files and directories we might recurse into
      remote_root = join(remote_code_path, relpath(dirpath, local_code_path))
      for name in dirnames:
        conn.run(f"mkdir {join(remote_root, name)}")
      for file in filenames:
        conn.put(join(dirpath, file), join(remote_root, file))

  elif transfer_files == "rsync":
    exclude = [
      ".*",
      "docs",
      "t5x.egg-info",
      "tmp",
      "testdata",
      "wandb"
    ]
    args = ["rsync", "-rz", local_code_path, f"{conn.host}:{dirname(remote_code_path)}"]
    args += ["-e", "ssh -o StrictHostKeyChecking=no"]
    for ex in exclude:
      args += ["--exclude", ex]
    print(conn.host, f"Copy files over with rsync: {' '.join(args)}")
    subprocess.run(args, check=True)
  elif transfer_files == "none":
    pass
  else:
    raise NotImplementedError(transfer_files)

  if install_deps == "check":
    res = conn.run(f'python3 -c "import jax; import seqio; import jax.numpy as jnp"',
                   warn=True, hide=True)
    if res.exited == 0:
      print(conn.host, "python libraries found, skip install")
      install_deps = False
    elif res.exited == 1:
      print(conn.host, "python libraries not found, installing")
      install_deps = True
    else:
      raise RuntimeError()

  if install_deps:
    print(conn.host, f"Running installation script")
    install_script = join(remote_code_path, "tpu_startup_script.sh")
    conn.sudo(f'chmod +x {install_script}', hide=True)
    conn.run(install_script, hide=True)

  # This patches seqio, not sure if we still need this but I guess it stays for now
  conn.put(join(local_code_path, 'additional_file', 'utils.py'),
           f"/home/{tpu_home}/.local/lib/python3.8/site-packages/seqio/utils.py")


def run_on_connections(conns: List[Connection], username, command, install_deps=True, disown=True):
  """Given set of connections to a TPU, run a command or commands on the TPU"""

  if isinstance(command, str):
    print(f"Will launch command: {command} on {len(conns)} connecton")
  else:
    print(f"Launched {len(command)} commands")
    for name, sub_command in command:
      print(f"{name}: {sub_command}")

  print("Setting up instances....")
  if len(conns) == 1:
    setup_tpu(conns[0], username, "rsync", install_deps)
  else:
    with ThreadPool(processes=len(conns)) as p:
      p.starmap(setup_tpu, [(c, username, "rsync", install_deps) for c in conns])

  def _run(conn, _command, _disown=True, out_name=None):
    if out_name is None:
      out_name = "out.txt"
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    with conn.cd('~/t5x'):
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


def get_connections(host, zone):
  key_path = expanduser('~/.ssh/google_compute_engine')

  out = subprocess.getoutput(f"gcloud alpha compute tpus tpu-vm describe --zone {zone} {host} --format json")
  out = json.loads(out)
  ips = [x["accessConfig"]["externalIp"] for x in out["networkEndpoints"]]
  print(f"Identified {ips} ips for host {host}")

  # This will (sometimes?) take care of some know-host issue that would otherwise prevent us
  # from ssh-ing in normally
  # Might be some ssh things we could do to fix this in a better way
  print(f"Testing connection with gcloud ssh....")
  exit_code = os.system('gcloud alpha compute tpus tpu-vm ssh {} --zone {} --command="echo gcloud connected"'.format(host, zone))
  if exit_code != 0:
    raise ValueError(f"gcloud connection failed, host {host} might be not be reachable")

  conns = [Connection(h, connect_kwargs={"key_filename": key_path}) for h in ips]
  return conns
