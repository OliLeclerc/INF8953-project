DOCKER_NAME := inf8953_ars_image
CONTAINER_NAME := inf8953_ars_container

ENV_NAME := HalfCheetah-v2
SEED:= 1904016 1954056 1952752 # can be a single seed or multiple seed: 10 346345 999
V2:= #--is_v2 #leave empty if you want v1 else --is_v2
t:= #--t #leave empty if you want base version else --t
rs:= # --rs if you want to run over a 100 random seed, else leave blank and it will take the given seeds
RENDER:= --r # if you want to render or not.

build:
	docker build -t $(DOCKER_NAME) .

run:
	docker run --name $(CONTAINER_NAME) -t $(DOCKER_NAME) python inf8953_project.py $(ENV_NAME) $(SEED) $(V2) $(RENDER) $(t) $(rs)
	docker cp $(CONTAINER_NAME):inf8953_projet/outputs ../
	docker rm $(CONTAINER_NAME)
	
clean:
	docker image rm $(DOCKER_NAME)
	
