# Step-by-Step Local Deployment of BEIR Code

Lixiao Yang \
7/30/2024

This notebook illustrates the steps I conducted to setup BEIR locally and on BVM96 machine.\
Note: this instruction is based on Windows OS, deployment on MacOS / Linux can be slightly different.

## 1. Create conda environment
Create a separate conda environment (use Python version 3.6 or 3.7):
https://stackoverflow.com/questions/57940783/how-to-install-the-specific-version-of-python-with-anaconda
```cmd
conda create --name <env> python=3.7 --channel conda-forge
```
Activate the environment:
https://coderefinery.github.io/installation/conda-environment/
```cmd
conda activate <env>
```

**For `CondaOpenSSLError:**\
Getting OpenSSL working:
https://community.anaconda.cloud/t/getting-openssl-working/51512

## 2. BEIR environment configuration
Follow instructions on [BEIR Repository](https://github.com/beir-cellar/beir)

**For Windows OS, the `pytrec_eval` is [not supported](https://github.com/cvangysel/pytrec_eval/issues/1), following error will occur:**
```cmd
      ValueError: path '/Users/cvangysel/Projects/pytrec_eval/trec_eval/convert_zscores.c' cannot be absolute
      [end of output]

  note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for pytrec_eval
  Running setup.py clean for pytrec_eval
  Building wheel for sentence-transformers (setup.py) ... done
  Created wheel for sentence-transformers: filename=sentence_transformers-2.2.2-py3-none-any.whl size=125930 sha256=4dd51fe3818851276b85d78c88972a59b2694e72d922a7fee8e7d5b7af72c882
  Stored in directory: c:\users\24075\appdata\local\pip\cache\wheels\bf\06\fb\d59c1e5bd1dac7f6cf61ec0036cc3a10ab8fecaa6b2c3d3ee9
Successfully built sentence-transformers
Failed to build pytrec_eval
ERROR: Could not build wheels for pytrec_eval, which is required to install pyproject.toml-based projects
```
Refer to the following solution: https://stackoverflow.com/questions/48493505/

> Hello,
>
> This is a known bug on Windows machines. The fix is to copy the two below DLL’s from the ‘bin’ folder to the 'DLL’s folder:
>
>On Windows machines the anaconda installation directory will reside under one of the following, depending on whether you proceed with a ‘system-wide’ or ‘per-user’ install:
>
>‘C:\ProgramData\anaconda3’ (system-wide install)
>‘C:\Users<your_user_name>\anaconda3’ (‘per-user’ install)
>
>Under this directory there will be a ‘Library\bin’ folder.
>
>You can search, copy and paste using ‘Windows Explorer’.
>
>Copy and paste the files:
>libcrypto-1_1-x64.dll
>libssl-1_1-x64.dll
>
>from this folder
>
>into the directory:
>
>‘C:\Users<your_login_name\anaconda3\DLLs’ folder.
>
>This should fix the problem.

Pull the BEIR repository:
```cmd
git clone https://github.com/beir-cellar/beir.git
cd beir
pip install -e .
```

Test with the scripts provided, also included [here](./beit_test.py), also log the result:
```cmd
python beit_test.py | tee beir_test_output.txt
```
See the result at [here](./beir_test_output.txt)

## 3. Local Elasticsearch Deployment using Docker

https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html

Build a new container for elastic:
```cmd
docker run --name local_es --net elastic -p 9200:9200 -it -m 3GB docker.elastic.co/elasticsearch/elasticsearch:8.14.3
```

```cmd
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Elasticsearch security features have been automatically configured!
✅ Authentication is enabled and cluster connections are encrypted.

ℹ️  Password for the elastic user (reset with `bin/elasticsearch-reset-password -u elastic`):
  [hidden]

ℹ️  HTTP CA certificate SHA-256 fingerprint:
  [hidden]

ℹ️  Configure Kibana to use this cluster:
• Run Kibana and click the configuration link in the terminal when Kibana starts.
• Copy the following enrollment token and paste it into Kibana in your browser (valid for the next 30 minutes):
  [hidden]

ℹ️ Configure other nodes to join this cluster:
• Copy the following enrollment token and start new Elasticsearch nodes with `bin/elasticsearch --enrollment-token <token>` (valid for the next 30 minutes):
  [hidden]

  If you're running in Docker, copy the enrollment token and run:
  `docker run -e "ENROLLMENT_TOKEN=<token>" docker.elastic.co/elasticsearch/elasticsearch:8.14.3`
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

For fixing potential connection password issue:
```cmd
curl.exe -u elastic --cacert c:/Users/24075/http_ca.crt https://localhost:9200 --insecure
```

Create elastic index:
```cmd
curl.exe -u elastic -X PUT "https://localhost:9200/reranking" --insecure
Enter host password for user 'elastic':
{"acknowledged":true,"shards_acknowledged":true,"index":"reranking"}
```

## 4. Elastic Deployment on BVM96
Enter the root machine:
```cmd
sudo -i
```

### Docker Installation
https://docs.docker.com/engine/install/ubuntu/
Set up Docker's apt repository.
```cmd
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
```
Install Docker packages
```cmd
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

### Elasticsearch Installation in Docker's Container
Follow previous part for local deployment.

List all running and stopped Docker containers to identify the existing container.
```cmd
docker ps -a
```
Start the Elasticsearch container:
```cmd
docker run --name es01 --net elastic -p 9200:9200 -it -m 60GB docker.elastic.co/elasticsearch/elasticsearch:8.14.3

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Elasticsearch security features have been automatically configured!
✅ Authentication is enabled and cluster connections are encrypted.

ℹ️  Password for the elastic user (reset with `bin/elasticsearch-reset-password -u elastic`):
  Q*0GbXCuvi9-Fx53U1Ii

ℹ️  HTTP CA certificate SHA-256 fingerprint:
  e203022e50539f99807c4c8d63f4b650bc0ff664821f16d02557b97dcd9cbe8d

ℹ️  Configure Kibana to use this cluster:
• Run Kibana and click the configuration link in the terminal when Kibana starts.
• Copy the following enrollment token and paste it into Kibana in your browser (valid for the next 30 minutes):
  eyJ2ZXIiOiI4LjE0LjAiLCJhZHIiOlsiMTcyLjE4LjAuMjo5MjAwIl0sImZnciI6ImUyMDMwMjJlNTA1MzlmOTk4MDdjNGM4ZDYzZjRiNjUwYmMwZmY2NjQ4MjFmMTZkMDI1NTdiOTdkY2Q5Y2JlOGQiLCJrZXkiOiJSLVA4NEpBQlR5LTVNQVlRR1dwODotUndVUlNsRlJNaUU2UHlTQzk2ZF9nIn0=

ℹ️ Configure other nodes to join this cluster:
• Copy the following enrollment token and start new Elasticsearch nodes with `bin/elasticsearch --enrollment-token <token>` (valid for the next 30 minutes):
  eyJ2ZXIiOiI4LjE0LjAiLCJhZHIiOlsiMTcyLjE4LjAuMjo5MjAwIl0sImZnciI6ImUyMDMwMjJlNTA1MzlmOTk4MDdjNGM4ZDYzZjRiNjUwYmMwZmY2NjQ4MjFmMTZkMDI1NTdiOTdkY2Q5Y2JlOGQiLCJrZXkiOiJSdVA4NEpBQlR5LTVNQVlRR1dwODp5eTJSYjZOTVF0dWNzY1JvZHhIUVZnIn0=

  If you're running in Docker, copy the enrollment token and run:
  `docker run -e "ENROLLMENT_TOKEN=<token>" docker.elastic.co/elasticsearch/elasticsearch:8.14.3`
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
Export the password, set as environment variable:
```cmd
export ELASTIC_PASSWORD="your_password"
```

Copy the http_ca.crt SSL certificate from the container to local machine.
```cmd
docker cp es01:/usr/share/elasticsearch/config/certs/http_ca.crt .
```
Make a REST API call to Elasticsearch to ensure the Elasticsearch container is running.
```cmd
curl --cacert http_ca.crt -u elastic:$ELASTIC_PASSWORD https://localhost:9200
```
Switch to current user.
```cmd
su - ly364
```

### Install and Configure Anaconda in Root
Version chose: `Anaconda3-2024.06-1-Linux-x86_64.sh`
```cmd
curl -O https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
```
Install anaconda:
```cmd
bash Anaconda3-2024.06-1-Linux-x86_64.sh
```
Activate conda:
```cmd
root@bvm96:~# eval "$(/root/anaconda3/bin/conda shell.bash hook)"

(base) root@bvm96:~# conda config --set auto_activate_base true
```

## 5. Workflow of Running the Scripts on the Remote Machine

### Start the services (root permission required)
List all running and stopped Docker containers to identify the existing container. DO NOT STOP OTHER CONTAINERS IF USING THE SAME PORTS (9200/9300)!
```cmd
(base) root@bvm96:~# docker ps -a
CONTAINER ID   IMAGE                                                  COMMAND                  CREATED       STATUS                        PORTS     NAMES
b69661e5fb1a   docker.elastic.co/elasticsearch/elasticsearch:8.14.3   "/bin/tini -- /usr/l…"   7 hours ago   Exited (143) 27 seconds ago             deep_delight
659ce14ba75d   hello-world                                            "/hello"                 8 hours ago   Exited (0) 8 hours ago                  affectionate_antonelli
0ea3059bcefa   hello-world                                            "/hello"                 8 hours ago   Exited (0) 8 hours ago                  trusting_mcnulty
```
Start the container `deep_delight'
```cmd
docker start es01
```
Check the Status of the Container:
```cmd
docker ps -a | grep es01
```
Start Elasticsearch Service Within the Container:
```cmd
docker run --name es01 --net elastic -p 9201:9200 -it -m 62GB docker.elastic.co/elasticsearch/elasticsearch:8.14.3

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Elasticsearch security features have been automatically configured!
✅ Authentication is enabled and cluster connections are encrypted.

ℹ️  Password for the elastic user (reset with `bin/elasticsearch-reset-password -u elastic`):
  2udpV0nS-adWekOyiTiz

ℹ️  HTTP CA certificate SHA-256 fingerprint:
  d5e5f00d4f7a5477a41691ad7ee5b92a1400ec0d8ea9f3022ce21759619e0bb5

ℹ️  Configure Kibana to use this cluster:
• Run Kibana and click the configuration link in the terminal when Kibana starts.
• Copy the following enrollment token and paste it into Kibana in your browser (valid for the next 30 minutes):
  eyJ2ZXIiOiI4LjE0LjAiLCJhZHIiOlsiMTcyLjE4LjAuMjo5MjAwIl0sImZnciI6ImQ1ZTVmMDBkNGY3YTU0NzdhNDE2OTFhZDdlZTViOTJhMTQwMGVjMGQ4ZWE5ZjMwMjJjZTIxNzU5NjE5ZTBiYjUiLCJrZXkiOiIyVXpNNHBBQm1OWGJPck53TUlzRzpKVF9LdmV4UFNTLVBqUnF5amZEXzVBIn0=

ℹ️ Configure other nodes to join this cluster:
• Copy the following enrollment token and start new Elasticsearch nodes with `bin/elasticsearch --enrollment-token <token>` (valid for the next 30 minutes):
  eyJ2ZXIiOiI4LjE0LjAiLCJhZHIiOlsiMTcyLjE4LjAuMjo5MjAwIl0sImZnciI6ImQ1ZTVmMDBkNGY3YTU0NzdhNDE2OTFhZDdlZTViOTJhMTQwMGVjMGQ4ZWE5ZjMwMjJjZTIxNzU5NjE5ZTBiYjUiLCJrZXkiOiIxMHpNNHBBQm1OWGJPck53TUlzRjpVYXN6Q3ZSX1FqRzJTYmRZdEF6TEp3In0=

  If you're running in Docker, copy the enrollment token and run:
  `docker run -e "ENROLLMENT_TOKEN=<token>" docker.elastic.co/elasticsearch/elasticsearch:8.14.3`
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```
Start another cmd, start a new index within the elastic
```cmd
curl -u elastic:$ELASTIC_PASSWORD -X PUT "https://localhost:9201/deepdelight" --insecure
```

### Local code running
1. Switch to current user and activate `beir` environment
```cmd
su - ly364
conda activate beir
```
2. Modify the password, port, and index information in the running script, in this case:
```python
elastic_password = "2udpV0nS-adWekOyiTiz"
hostname = "https://localhost:9201"
index_name = "deepdelight"
```
3. Run the code...
```cmd
python ... | tee ...
```

### End the services (root permission required)
End docker and remove the container
```cmd
sudo -i
docker stop es01
docker rm es01
```

## Complete packages list
```cmd
(beir_py) PS C:\Users\24075\beir> conda list
# packages in environment at C:\Users\24075\.conda\envs\beir_py:
#
# Name                    Version                   Build  Channel
aiohttp                   3.8.6                    pypi_0    pypi
aiosignal                 1.3.1                    pypi_0    pypi
async-timeout             4.0.3                    pypi_0    pypi
asynctest                 0.13.0                   pypi_0    pypi
attrs                     23.2.0                   pypi_0    pypi
backcall                  0.2.0              pyhd3eb1b0_0
beir                      2.0.0                     dev_0    <develop>
blas                      1.0                         mkl
ca-certificates           2024.3.11            haa95532_0
certifi                   2024.7.4                 pypi_0    pypi
charset-normalizer        3.3.2                    pypi_0    pypi
click                     8.1.7                    pypi_0    pypi
colorama                  0.4.6            py37haa95532_0
datasets                  2.13.2                   pypi_0    pypi
debugpy                   1.5.1            py37hd77b12b_0
decorator                 5.1.1              pyhd3eb1b0_0
dill                      0.3.6                    pypi_0    pypi
elasticsearch             7.9.1                    pypi_0    pypi
entrypoints               0.4              py37haa95532_0
faiss-cpu                 1.7.4                    pypi_0    pypi
filelock                  3.12.2                   pypi_0    pypi
frozenlist                1.3.3                    pypi_0    pypi
fsspec                    2023.1.0                 pypi_0    pypi
git                       2.40.1               haa95532_4
huggingface-hub           0.16.4                   pypi_0    pypi
idna                      3.7                      pypi_0    pypi
importlib-metadata        6.7.0                    pypi_0    pypi
intel-openmp              2021.4.0          haa95532_3556
ipykernel                 6.15.2           py37haa95532_0
ipython                   7.31.1           py37haa95532_1
jedi                      0.18.1           py37haa95532_1
joblib                    1.3.2                    pypi_0    pypi
jupyter_client            7.4.9            py37haa95532_0
jupyter_core              4.11.2           py37haa95532_0
libsodium                 1.0.18               h62dcd97_0
libsqlite                 3.46.0               h2466b09_0    conda-forge
matplotlib-inline         0.1.6            py37haa95532_0
mkl                       2021.4.0           haa95532_640
mkl-service               2.4.0            py37h2bbff1b_0
mkl_fft                   1.3.1            py37h277e83a_0
mkl_random                1.2.2            py37hf11a4ad_0
multidict                 6.0.5                    pypi_0    pypi
multiprocess              0.70.14                  pypi_0    pypi
nest-asyncio              1.5.6            py37haa95532_0
nltk                      3.8.1                    pypi_0    pypi
numpy                     1.21.5           py37h7a0a035_3
numpy-base                1.21.5           py37hca35cd5_3
openssl                   3.0.14               h827c3e9_0
packaging                 24.0                     pypi_0    pypi
pandas                    1.3.5                    pypi_0    pypi
parso                     0.8.3              pyhd3eb1b0_0
pickleshare               0.7.5           pyhd3eb1b0_1003
pillow                    9.5.0                    pypi_0    pypi
pip                       24.0               pyhd8ed1ab_0    conda-forge
prompt-toolkit            3.0.36           py37haa95532_0
psutil                    5.9.0            py37h2bbff1b_0
pyarrow                   12.0.1                   pypi_0    pypi
pygments                  2.11.2             pyhd3eb1b0_0
python                    3.7.12          h900ac77_100_cpython    conda-forge
python-dateutil           2.9.0.post0              pypi_0    pypi
pytrec-eval               0.5                      pypi_0    pypi
pytz                      2024.1                   pypi_0    pypi
pywin32                   305              py37h2bbff1b_0
pyyaml                    6.0.1                    pypi_0    pypi
pyzmq                     23.2.0           py37hd77b12b_0
regex                     2024.4.16                pypi_0    pypi
requests                  2.31.0                   pypi_0    pypi
safetensors               0.4.3                    pypi_0    pypi
scikit-learn              1.0.2                    pypi_0    pypi
scipy                     1.7.3                    pypi_0    pypi
sentence-transformers     2.2.2                    pypi_0    pypi
sentencepiece             0.2.0                    pypi_0    pypi
setuptools                65.6.3             pyhd8ed1ab_0    conda-forge
six                       1.16.0             pyhd3eb1b0_1
sqlite                    3.46.0               h2466b09_0    conda-forge
threadpoolctl             3.1.0                    pypi_0    pypi
tokenizers                0.13.3                   pypi_0    pypi
torch                     1.13.1                   pypi_0    pypi
torchvision               0.14.1                   pypi_0    pypi
tornado                   6.2              py37h2bbff1b_0
tqdm                      4.66.4                   pypi_0    pypi
traitlets                 5.7.1            py37haa95532_0
transformers              4.30.2                   pypi_0    pypi
typing-extensions         4.7.1                    pypi_0    pypi
ucrt                      10.0.22621.0         h57928b3_0    conda-forge
urllib3                   2.0.7                    pypi_0    pypi
vc                        14.3                h8a93ad2_20    conda-forge
vc14_runtime              14.40.33810         ha82c5b3_20    conda-forge
vs2015_runtime            14.40.33810         h3bf8584_20    conda-forge
wcwidth                   0.2.5              pyhd3eb1b0_0
wheel                     0.34.2                   py37_0    conda-forge
xxhash                    3.4.1                    pypi_0    pypi
yarl                      1.9.4                    pypi_0    pypi
zeromq                    4.3.4                hd77b12b_0
zipp                      3.15.0                   pypi_0    pypi
```
