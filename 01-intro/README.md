# MLOPS Intro
## Launch AWS Instance
### Configuration
Amazon Machine Image: `Ubuntu`   

Instance Type: `t2.large` or `t2.xlarge`   

Key Pairs: 
- Key Pair Type: `RSA`
- Private Key File Format: `.pem`

Configure Storage: `30 GiB`

### Login via Local Machine
Save the following setting in `./.ssh/config`
```
Host mlops-zoomcamp
    HostName <public_ipv4_address>
    User ubuntu
    IdentityFile ~/.ssh/aws.pem
    StrictHostKeyChecking no 
```
Login to AWS VM 
```
ssh mlops-zoomcamp
```

### Download Anaconda
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
bash Anaconda3-2024.10-1-Linux-x86_64.sh
```

### Activate Conda Environment
```bash
eval "$(/home/ubuntu/anaconda3/bin/conda shell.bash hook)"
conda init
# Deactivate Conda
conda config --set auto_activate_base false
```

### Download Docker & Docker Compose
```bash
# Download docker
sudo apt update
sudo apt install docker.io

# Download docker compose
wget https://github.com/docker/compose/releases/download/v2.36.1/docker-compose-linux-x86_64 -O docker-compose
chmod +x docker-compose

# Add docker-compose to PATH in .bashrc
export PATH="${HOME}/soft:${PATH}"
source .bashrc
```

### Docker without Sudo
```bash
sudo groupadd docker
sudo usermod -aG docker $USER
```

### Connect VM to VSCode
First to install an extension `Remote-SSH`, then open the remote window and click `Connect to Host`. Select host name `mlops-zoomcamp` on pop-up (The pop-up lists `HostName` in `.ssh/config`).


### Training Model
#### Lambda
`lambda` is a simple version of `def function`. 
```python
# Lambda
df['duration'] = df['duration'].apply(lambda time: time.total_seconds() / 60)

# Def Function
def lambda(time):
    return time.total_seconds() / 60

lambda(df['duration'])
```
- `apply()` & `lambda` act like a  defined function
- `time` represents variable
- `df['duration]` is an instance. 

#### One-Hot Encoding
```python
from sklearn.feature_extraction import DictVectorizer
dv = DictVectorizer()
dict = df.to_dict(orient='records')
dv.fit_transform(dict) #DictVectorizer only accpets python dictionary
```

#### Linear Regression
LR has multiple versions - Linear Regression, Lasso, Ridge.
```python
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

lr = LinearRegression() 
lr.fit(X, y)

y_pred = lr.predict(X)
```

#### Model Evaluation
```python
from sklearn.metrics import mean_squared_error
mean_squared_error(y_actual, y_pred, squared=False)
```

#### Pack Model
```python
import pickle
with open('models/lin_reg.bin', 'wb') as f_out:
    pickle.dump((dv, lr), f_out)
```