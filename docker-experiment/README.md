## Docker Setup

The container setup is as follows:
- A Server container that runs a FastAPI server.
- An Ingestion container that handles pulling data from two APIs and cleaning/putting them together.
    - After the data pull is finished and the flight and weather data are merged together, a POST request is made to the FastAPI server to enable pushing the output of the ingestion operation to the server.
- A preprocessing container that handles the necessary data pre-processing tasks.
    - This container makes a GET request to the FastAPI server, receiving the ingestion output.
    - It performs the necessary pre-processing tasks, and pushes the output back to the server with a POST request.
- A modeling container that trains the data.
    - This container pulls the data from the FastAPI server through a GET request.
    - This container then makes predictions which is then transferred back to the FastAPI server with a POST request.

How does it work?

- All of these microservices are orchestrated using a docker compose file in the docker-experiment/ folder.
- There is a custom polling mechanism code that is written in the server source code - this ensures that each microservice waits to receive the finished output from its upstream dependencies before attempting to run its application code.
