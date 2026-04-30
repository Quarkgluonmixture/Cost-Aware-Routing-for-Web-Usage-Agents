We are happy to confirm that your account to use the Research Computing Myriad
HPC cluster is now active. You should be able to log in within 5 minutes of
receiving this email.

Please find below some information to help get you get started in your use of
the system.

GETTING HELP

Information to help you get started in using Myriad is available at

https://www.rc.ucl.ac.uk/docs/

including a user guide covering all of our systems.

ANNOUNCEMENTS

Emails relating to planned outages, service changes etc will be sent to the
myriad-users@ucl.ac.uk email list. You have been subscribed to this
list using the email address associated with your main UCL userid - please
make sure that you read all notices sent to this address promptly and
observe the requests/guidelines they contain.
If you use a different email address for most of your correspondence, it is
a condition of your account that you configure your UCL email account to
have email redirected to an address of your choosing.

Please see https://www.ucl.ac.uk/isd/how-to/set-forwarding-using-outlook-web-access-owa
for further information on email forwarding.

If you have any queries relating to this information please email the
support address rc-support@ucl.ac.uk.

Your request for Research Computing resources has been completed.

We have a number of services available for users, and the approval process automatically recommends services based on the list of work types on the application form.


This has recommended: <QuerySet [<Service: Myriad>]>


 \* The Myriad service is a cluster designed to be most suitable for serial work, including large numbers of serial jobs, and multi-threaded jobs (using e.g. OpenMP). It also includes a small number of GPUs for development or testing work. We recommend using Myriad for most general-purpose use, and any use not explicitly covered by another service.

 \* The Kathleen service is a large high-performance cluster, set-up to be most suitable for multi-node jobs (using e.g. MPI). We recommend using Kathleen if you intend to use more than 36 cores per job.

 More information on these services is available from the Research Computing website, at: https://www.ucl.ac.uk/advanced-research-computing/platforms/research-computing-platforms

We create accounts on the Myriad service for all applying researchers regardless of work type.

# UCL Research Computing Renewal Request

NameJiaming WeiEmailjiaming.wei.25@ucl.ac.ukUPIJWEIC63DepartmentDept of Computer ScienceUCL useriducab352

# Myriad[§](https://www.rc.ucl.ac.uk/docs/Clusters/Myriad/#myriad)

Myriad is designed for high I/O, high throughput jobs that will run within a single node rather than multi-node parallel jobs.

## Accounts[§](https://www.rc.ucl.ac.uk/docs/Clusters/Myriad/#accounts)

Myriad accounts can be applied for via the [Research Computing sign up process](https://www.rc.ucl.ac.uk/docs/Account_Services/).

As Myriad is our most general-purpose system, everyone who signs up for a Research Computing account is given access to Myriad.

## Logging in[§](https://www.rc.ucl.ac.uk/docs/Clusters/Myriad/#logging-in)

You will use your UCL username and password to ssh in to Myriad.

```
ssh uccaxxx@myriad.rc.ucl.ac.uk
```

If using PuTTY, put `myriad.rc.ucl.ac.uk` as the hostname and your seven-character username (with no @ after) as the username when logging in, eg. `uccaxxx`. When entering your password in PuTTY no characters or bulletpoints will show on screen - this is normal.

If you are outside the UCL firewall you will need to follow the instructions for [Logging in from outside the UCL firewall](https://www.rc.ucl.ac.uk/docs/howto/#logging-in-from-outside-the-ucl-firewall).

The login nodes allow you to manage your files, compile code and submit jobs. Very short (< 15 mins) and non-resource-intensive software tests can be run on the login nodes, but anything more should be submitted as a job.

### Logging in to a specific node[§](https://www.rc.ucl.ac.uk/docs/Clusters/Myriad/#logging-in-to-a-specific-node)

You can access a specific Myriad login node with:

```
ssh uccaxxx@login12.myriad.rc.ucl.ac.uk
ssh uccaxxx@login13.myriad.rc.ucl.ac.uk
```

The main address will redirect you on to either one of them.

## Copying data onto Myriad[§](https://www.rc.ucl.ac.uk/docs/Clusters/Myriad/#copying-data-onto-myriad)

You will need to use an SCP or SFTP client to copy data onto Myriad. Please refer to the page on [How do I transfer data onto the system?](https://www.rc.ucl.ac.uk/docs/howto/#how-do-i-transfer-data-onto-the-system)

## Quotas[§](https://www.rc.ucl.ac.uk/docs/Clusters/Myriad/#quotas)

The default quota on Myriad is 1TB for home (which is also considered scratch).

The "hard limit" number means that once you reach it, you will no longer be able to write more data. Keep an eye on your quota, as this will cause jobs to fail if they cannot create their .o or .e files at the start, or their output files partway through.

You can check your quota on Myriad by running:

```
gquota
```

which will give you output similar to this:

```
Current Usage: 108.7GiB
Soft Limit: 1024GiB
Hard Limit: 1024GiB
```

You can apply for quota increases using the form at [Additional Resource Requests](https://www.rc.ucl.ac.uk/docs/Additional_Resource_Requests/).

Here are some tips for [managing your quota](https://www.rc.ucl.ac.uk/docs/howto/#managing-your-quota) and finding where space is being used.

## Job sizes[§](https://www.rc.ucl.ac.uk/docs/Clusters/Myriad/#job-sizes)

| Cores                        | Max wallclock |
| :--------------------------- | :------------ |
| 1                            | 72hrs         |
| 2 to 36                      | 48hrs         |
| 2 to 48 (UV type nodes only) | 48hrs         |
| 2 to 64 (T type nodes only)  | 48hrs         |

[Interactive jobs](https://www.rc.ucl.ac.uk/docs/Interactive_Jobs/) run with `qrsh` have the same maximum wallclock time as other jobs.

## Node types[§](https://www.rc.ucl.ac.uk/docs/Clusters/Myriad/#node-types)

Myriad contains three main node types: standard compute nodes, high memory nodes and GPU nodes. As new nodes as added over time with slightly newer processor variants, new letters are added.

| Type | Cores per node       | RAM per node | Max usable RAM per node                | tmpfs | Nodes |
| :--- | :------------------- | :----------- | :------------------------------------- | :---- | :---- |
| D    | 36                   | 192GB        | 160GB (or 36 cores at 4.4G per core)   | 1500G | 342   |
| I,B  | 36                   | 1.5TB        | 1483GB (or 36 cores at 41.1G per core) | 1500G | 17    |
| E,F  | 36 + 2 V100 GPUs     | 192GB        | 160GB (or 36 cores at 4.4G per core)   | 1500G | 19    |
| L    | 36 + 4 A100 40G GPUs | 192GB        | 160GB (or 36 cores at 4.4G per core)   | 1500G | 6     |
| T    | 64                   | 768GB        | 755GB (or 64 cores at 11.7G per core)  | 420G  | 6     |
| U    | 48 + 4 A100 80G GPUs | 256GB        | 223GB (or 48 cores at 4.6G per core)   | 1700G | 1     |
| V    | 48 + 4 A100 80G GPUs | 256GB        | 223GB (or 48 cores at 4.6G per core)   | 1800G | 2     |

You can tell the type of a node by its name: type D nodes are named `node-d00a-001` etc.

Here are the processors each node type has:

- F, I: Intel(R) Xeon(R) Gold 6140 CPU @ 2.30GHz
- B, D, E, L: Intel(R) Xeon(R) Gold 6240 CPU @ 2.60GHz
- T: AMD EPYC 9554P 64C 360W 3.1GHz (64 cores)
- U: Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz
- V: Intel(R) Xeon(R) Gold 6342 CPU @ 2.80GHz

(If you ever need to check this, you can include `cat /proc/cpuinfo` in your jobscript so you get it in your job's .o file for the exact node your job ran on. You will get an entry for every core).

## GPUs[§](https://www.rc.ucl.ac.uk/docs/Clusters/Myriad/#gpus)

Myriad has several types of GPU nodes: E, F, L, U and V.

- U-type and V-type each have four NVIDIA 80G A100s. (Compute Capability 80). The CPUs and disk are slightly different.
- L-type nodes each have four NVIDIA 40G A100s. (Compute Capability 80)
- F-type and E-type nodes each have two NVIDIA Tesla V100s. The CPUs are slightly different on the different letters, see above. (Compute Capability 70)

You can include `nvidia-smi` in your jobscript to get information about the GPU your job ran on.

### Compute Capability[§](https://www.rc.ucl.ac.uk/docs/Clusters/Myriad/#compute-capability)

[Compute Capability](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-generations) is how NVIDIA categorises its generations of GPU architectures. When code is compiled, it targets one or multiple of these and so it may only be able to run on GPUs of a specific Compute Capability.

If you get an error like this:

```
CUDA runtime implicit initialization on GPU:0 failed. Status: device kernel image is invalid
```

then the software you are running does not support the Compute Capability of the GPU you tried to run it on, and you probably need a newer version.

### Requesting multiple and specific types of GPU[§](https://www.rc.ucl.ac.uk/docs/Clusters/Myriad/#requesting-multiple-and-specific-types-of-gpu)

You can request a number of GPUs by adding them as a resource request to your jobscript:

```
# For 1 GPU
#$ -l gpu=1

# For 2 GPUs
#$ -l gpu=2

# For 4 GPUs
#$ -l gpu=4
```

If you ask for one or two GPUs your job can run on any type of GPU since it can fit on any of the nodetypes. If you ask for four, it can only be a node that has four. If you need to specify one node type over the others because you need a particular Compute Capability, add a request for that type of node to your jobscript:

```
# request a V100 node only
#$ -ac allow=EF

# request an A100 node only
#$ -ac allow=L
```

The [GPU nodes](https://www.rc.ucl.ac.uk/docs/Supplementary/GPU_Nodes/) page has some sample code for running GPU jobs if you need a test example.

### Tensorflow[§](https://www.rc.ucl.ac.uk/docs/Clusters/Myriad/#tensorflow)

Tensorflow is installed: type `module avail tensorflow` to see the available versions.

Modules to load for the non-MKL GPU version:

```
module load python3/3.7
module load cuda/10.0.130/gnu-4.9.2
module load cudnn/7.4.2.24/cuda-10.0
module load tensorflow/2.0.0/gpu-py37
```

Modules to load the most recent version we have installed with GPU support (2.11.0):

```
module unload compilers mpi gcc-libs
module load gcc-libs/10.2.0
module load python/3.9.6-gnu-10.2.0
module load cuda/11.2.0/gnu-10.2.0
module load cudnn/8.1.0.77/cuda-11.2
module load tensorflow/2.11.0/gpu
```

### PyTorch[§](https://www.rc.ucl.ac.uk/docs/Clusters/Myriad/#pytorch)

PyTorch is installed: type `module avail pytorch` to see the versions available.

Modules to load the most recent release we have installed (May 2022) are:

```
module unload compilers mpi gcc-libs
module load gcc-libs/10.2.0
module load python3/3.9-gnu-10.2.0
module load cuda/11.3.1/gnu-10.2.0
module load cudnn/8.2.1.32/cuda-11.3
module load pytorch/1.11.0/gpu
```

If you want the CPU only version then use:

```
module unload compilers mpi gcc-libs
module load gcc-libs/10.2.0
module load python3/3.9-gnu-10.2.0
module load pytorch/1.11.0/cpu
```