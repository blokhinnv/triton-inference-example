# Simple example of using the Triton Inference Server

Thanks to Ozon and the authors of the Triton course for inviting me to take the course and inspiring me to explore this new technology.


1. Navigate to the directory containing docker-compose.yml and start the Triton container:
```bash
docker-compose up -d
```

2. After the container starts, access its shell (where N is the container ID):
`docker exec -it N bash`

3. Create venv:
```
python -m venv .venv
pip install -r requirements.txt
```

4. Train the model: 
```
cd src/cifar10-cnn
python train.py
```

5. Save the trained model as a TorchScript model:
`python .\save.py`

6. For `cifar1-preprocessing` we need a custom env. Create it by calling `source /models/cifar10-preprocessing/create_env.sh` inside the triton container.

7. Start the Triton Inference Server with the following command:
`tritonserver --model-repository=/models`


8. Run the client: 
```
cd src/client
python client.py
```
