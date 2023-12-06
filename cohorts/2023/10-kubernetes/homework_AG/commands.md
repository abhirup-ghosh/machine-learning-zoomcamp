cp -r 05-deployment/homework/* 10-kubernetes/homework_AG/
cd 10-kubernetes/homework_AG/

docker build -t zoomcamp-model:hw10 .
docker run -it --rm -p 9696:9696 zoomcamp-model:hw10
python q6_test.py

brew install kind
kind --version

kind create cluster
kubectl cluster-info --context kind-kind
kubectl get pod
kubectl get deployment

kind load docker-image zoomcamp-model:hw10

<<Create deployment.yaml>>
kubectl apply -f deployment.yaml
kubectl get deployment
kubectl get pod

<<Create servicep.yaml>>