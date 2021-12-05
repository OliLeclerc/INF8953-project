# INF8953-project

docker build . -t inf8953
#set amount of cpu you want to give
docker run -i inf8953 --cpus-shares="2048"

#go on your docker container and run the file
inf8953_project.py --env="ENV MUJOCO"

#copy the results once finished with
docker cp <containerId>:/file/path/within/container /host/path/target