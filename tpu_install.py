import argparse
import functools
import multiprocessing.pool
import time

from tpu_run import install_dependencies, TPUCreator

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--tpu-pod-name', type=str, required=True)
  parser.add_argument('--tpu-size', type=int, required=True)
  args = parser.parse_args()

  if "central" in args.tpu_pod_name:
    zone = "us-central1-a"
  else:
    zone = "us-east1-d"
  tpu_creator = TPUCreator(name=args.tpu_pod_name, tpu_size=args.tpu_size, zone=zone)

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
    p.map(functools.partial(install_dependencies, run_sh="run.sh"), conns)
  time.sleep(30)
