DOCKER_NAME := inf8953_ars_image
CONTAINER_NAME := inf8953_ars_container

ENV_NAME := HalfCheetah-v2
SEED:= 10
V2:= --is_v2#leave empty if you want v1 else --is_v2
t:= --t#leave empty if you want base version else --t


build:
	docker build -t $(DOCKER_NAME) .

run:
	docker run --name $(CONTAINER_NAME) -t $(DOCKER_NAME) python inf8953_project.py $(ENV_NAME) $(SEED) $(V2) --r $(t)
	docker cp $(CONTAINER_NAME):inf8953_projet/outputs ../
	docker rm $(CONTAINER_NAME)
	
	
