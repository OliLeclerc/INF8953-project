# INF8953-project
To run this project, you must have docker installed: https://docs.docker.com/get-docker/
Then use the command: 
    make build 
to construct the image.


All of the differents arguments are defined at the top of the MakeFile
Once they are updated and saved, use command:
    make run 
to launch the experiment. Once finished, the container will be automatically deleted and the redsutls will be save at ../Outputs/, relative to this folder.