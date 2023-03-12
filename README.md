# CHAMPAGNE: Learning Real-world Conversation from Large-Scale Web Videos

This repo contains official JAX implementation of our paper "CHAMPAGNE: Learning Real-world Conversation from Large-Scale Web Videos"

## Installation

Note that all the commands in this document should be run in the commandline of
the TPU VM instance unless otherwise stated.

1.  Follow the
    [instructions](https://cloud.google.com/tpu/docs/jax-quickstart-tpu-vm#install_the_google_cloud_sdk)
    to set up a Google Cloud Platform (GCP) account and enable the Cloud TPU
    API.

    **Note:** While T5X works with GPU as well, we haven't heavily tested the
    GPU usage.

2.  Create a
    [Cloud TPU VM instance](https://cloud.google.com/blog/products/compute/introducing-cloud-tpu-vms)
    following
    [this instruction](https://cloud.google.com/tpu/docs/jax-quickstart-tpu-vm#create-vm).
    We recommend that you develop your workflow in a single v3-8 TPU (i.e.,
    `--accelerator-type=v3-8`) and scale up to pod slices once the pipeline is
    ready. In this README, we focus on using a single v3-8 TPU. See
    [here](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm) to
    learn more about TPU architectures.

3.  With Cloud TPU VMs, you ssh directly into the host machine of the TPU VM.
    You can install packages, run your code run, etc. in the host machine. Once
    the TPU instance is created, ssh into it with

    ```sh
    gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE}
    ```

    where `TPU_NAME` and `ZONE` are the name and the zone used in step 2.

4.  Install T5X and the dependencies.

    ```sh
    git clone --branch=main https://github.com/google-research/t5x
    cd t5x

    python3 -m pip install -e '.[tpu]' -f \
      https://storage.googleapis.com/jax-releases/libtpu_releases.html

    ```

5.  Create Google Cloud Storage (GCS) bucket to store the dataset and model
    checkpoints. To create a GCS bucket, see these
    [instructions](https://cloud.google.com/storage/docs/creating-buckets).

## Running Codes

### Setting Up TPU Instance before Running Code

    ```python
    # setting up TPU instance with 8 TPU cores
    python tpu_install.py --tpu-pod-name ${TPU_NAME} --tpu-size 8
    ```

### Run Script on TPU
    ```python
    # run script on TPU with 256 TPU cores
    python tpu_run.py --tpu-pod-name ${TPU_NAME} --tpu-size 256 --run-sh ${SCRIPT_SH} 
    ```

## References

- Unified-IO
