import argparse
import functools
import glob
import multiprocessing.pool
import os
import subprocess
from dataclasses import dataclass

import requests
from fabric import Connection

from settings import OWNER_NAME

multiprocessing.set_start_method("spawn")

import time
import json


@functools.lru_cache()
def get_bearer():
  return subprocess.check_output("gcloud auth print-access-token", shell=True).decode("utf-8").strip()


@functools.lru_cache()
def get_project():
  return subprocess.check_output("gcloud config list --format 'value(core.project)'", shell=True).decode(
    "utf-8").strip()


@dataclass
class TPUCreator:
  """
  Utility for creating TPUs and stuff
  """
  name: str
  tpu_size: int
  zone: str = 'us-east1-d'
  preemptible: bool = False
  network: str = ''
  subnetwork: str = ''
  version: str = 'v2-alpha'
  accelerator_type: str = 'v3'

  @property
  def base_url(self):
    # https://cloud.google.com/tpu/docs/reference/rest/v2alpha1/projects.locations.nodes/create
    return f'https://tpu.googleapis.com/v2alpha1/projects/{get_project()}/locations/{self.zone}/nodes'

  def check_tpu(self):
    response = requests.get(f'{self.base_url}/{self.name}',
                            headers={'Authorization': f'Bearer {get_bearer()}'})
    return response.json()

  def create_tpu(self):
    """
    Tries to create a TPU,
    :return: returns True if successful and False otherwise
    """
    if not os.path.expanduser('~/.ssh/google_compute_engine'):
      raise ValueError("Must create SSH keys in legacy mode with something like"
                       "ssh-keygen -m PEM -t rsa -b 4096 -C \"$(whoami)@$(hostname)\" -f ~/.ssh/google_compute_engine")

    try:
      status = self.check_tpu()

      if status["state"] not in ["CREATING", "READY"]:
        print("deleting TPU")
        self.delete_tpu()

        while True:
          try:
            print("deleting check: {}".format(self.check_tpu()["state"]), flush=True)
            time.sleep(1)
          except:
            break
    except:
      pass

    data = {
      "accelerator_type": f'{self.accelerator_type}-{self.tpu_size}',
      "runtime_version": f'{self.version}',
      "network_config": {"enable_external_ips": True, "network": self.network, "subnetwork": self.subnetwork},
      "tags": "unified_io",
    }

    if self.preemptible:
      data["schedulingConfig"] = {"preemptible": True}

    response = requests.post(self.base_url,
                             headers={'Authorization': f'Bearer {get_bearer()}',
                                      'Content-Type': 'application/json', },
                             params=(('node_id', self.name),), json=data)
    print(response.json())
    return response.status_code == 200

  def delete_tpu(self):
    response = requests.delete(f'{self.base_url}/{self.name}', headers={'Authorization': f'Bearer {get_bearer()}'})
    return response.json()

  def wait_until_tpu_ready(self):
    desired_state = {'state': 'READY', 'health': 'HEALTHY'}
    # desired_state = {'state': 'READY'}
    while True:
      ret = self.check_tpu()

      print(f"wait_until_tpu_ready check: {ret}", flush=True)

      if ("error" in ret) or (ret["state"] == "TERMINATED"):
        return False

      matches = True
      for k, expected_v in desired_state.items():
        if k not in ret:
          matches = False
          continue
        if ret[k] != expected_v:
          matches = False

      if matches:
        return True
      time.sleep(30)

  def get_connections(self):
    host = self.name
    zone = self.zone
    key_path = os.path.expanduser('~/.ssh/google_compute_engine')

    out = subprocess.getoutput(f"gcloud alpha compute tpus tpu-vm describe --zone {zone} {host} --format json")
    out = json.loads(out)
    ips = [x["accessConfig"]["externalIp"] for x in out["networkEndpoints"]]
    print(f"Identified {ips} ips for host {host}")

    # This will (sometimes?) take care of some know-host issue that would otherwise prevent us
    # from ssh-ing in normally
    # Might be some ssh things we could do to fix this in a better way
    print(f"Testing connection with gcloud ssh....")
    exit_code = os.system(
      'gcloud alpha compute tpus tpu-vm ssh {} --zone {} --command="echo gcloud connected"'.format(host, zone))
    if exit_code != 0:
      raise ValueError(f"gcloud connection failed, host {host} might be not be reachable")

    conns = [Connection(h, connect_kwargs={"key_filename": key_path}) for h in ips]
    return conns


def install_dependencies(conn, run_sh):
  """
  Upload all the code
  :param conn:
  :param address:
  :return:
  """
  try:
    conn.run(f'pkill -9 train.py')
  except Exception as e:
    print(e)

  try:
    conn.run(f'killall -9 screen')
  except Exception as e:
    print(e)

  print(f"Starting on {conn}", flush=True)
  conn.run('rm -rf *.py')
  conn.run('rm -rf *.json')
  conn.run('rm -rf screenlog.0')

  # copy credential for some error
  conn.run(f"mkdir /home/{OWNER_NAME}/.config/gcloud -p")
  conn.put(f'/home/{OWNER_NAME}/.config/gcloud/application_default_credentials.json', f'/home/{OWNER_NAME}/.config/gcloud')

  # conn.sudo('rm -rf *')
  local_code_path = os.path.expanduser('~/codes/unified-io/')
  # Copy files
  for i in glob.glob(os.path.join(local_code_path, '*.py')):
    conn.put(i, f'')

  for i in glob.glob(os.path.join(local_code_path, '*.md')):
    conn.put(i, f'')

  for ok_folder in ['t5x']:
    conn.sudo(f'rm -rf {ok_folder}')
    conn.run(f"mkdir {ok_folder} -p")
    for i in glob.glob(os.path.join(local_code_path, ok_folder, '*.*')):
      conn.put(i, f'{ok_folder}/')

  for ok_folder in ['metadata']:
    conn.sudo(f'rm -rf {ok_folder}')
    all_paths = glob.glob(os.path.join(local_code_path, ok_folder, '**', '*.*'), recursive=True)
    # get unique path for folder creation.
    folder_paths = set(['/'.join(p.split('/')[:-1]) for p in all_paths])
    for p in folder_paths:
      new_path = p.replace(local_code_path, '')
      conn.run(f"mkdir {new_path} -p")

    for p in all_paths:
      new_path = '/'.join(p.replace(local_code_path, '').split('/')[:-1])
      conn.put(p, new_path)

  for ok_folder in ['configs', 'examples']:
    conn.sudo(f'rm -rf {ok_folder}')
    all_paths = glob.glob(os.path.join(local_code_path, 't5x', ok_folder, '**', '*.*'), recursive=True)
    # get unique path for folder creation.
    folder_paths = set(['/'.join(p.split('/')[:-1]) for p in all_paths])
    for p in folder_paths:
      new_path = p.replace(local_code_path, '')
      conn.run(f"mkdir {new_path} -p")

    for p in all_paths:
      new_path = '/'.join(p.replace(local_code_path, '').split('/')[:-1])
      conn.put(p, new_path)

  conn.put(os.path.join(local_code_path, 'tpu_startup_script.sh'), "/tmp/startup.sh")
  conn.sudo('chmod +x /tmp/startup.sh', hide=True)
  conn.run('/tmp/startup.sh', hide=True)

  conn.put(os.path.join(local_code_path, run_sh), "run.sh")
  conn.sudo('chmod +x run.sh', hide=True)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--tpu-pod-name', type=str, required=True)
  parser.add_argument('--tpu-size', type=int, required=True)
  parser.add_argument('--run-sh', type=str, required=True)
  args = parser.parse_args()

  tpu_creator = TPUCreator(name=args.tpu_pod_name, tpu_size=args.tpu_size)

  while True:
    # tpu_creator.create_tpu()
    tpu_is_ready = tpu_creator.wait_until_tpu_ready()
    # info = tpu_creator.check_tpu()
    # if 'error' in info:
    if tpu_is_ready:
      break
    else:
      info = tpu_creator.check_tpu()
      print(f"\n~ERROR retrying: \n{info['error']}\n", flush=True)
      time.sleep(60 * 5)

  conns = tpu_creator.get_connections()

  with multiprocessing.pool.ThreadPool(processes=len(conns)) as p:
    p.map(functools.partial(install_dependencies, run_sh=args.run_sh), conns)
  time.sleep(30)


  def _run_pretrain(conn):
    with conn.cd(''):
      local_code_path = os.path.expanduser('~/codes/unified-io/')
      conn.put(os.path.join(local_code_path, 'additional_file', 'utils.py'),
               f"/home/{OWNER_NAME}/.local/lib/python3.8/site-packages/seqio/utils.py")
      conn.run(f'screen -d -m -L bash -c ./run.sh', pty=False)
      print('done')


  with multiprocessing.pool.ThreadPool(processes=len(conns)) as p:
    p.map(_run_pretrain, conns)
