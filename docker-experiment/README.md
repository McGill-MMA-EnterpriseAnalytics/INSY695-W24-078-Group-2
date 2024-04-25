## Docker Setup

The container setup is as follows:
- A Server container that runs a FastAPI server.
- An Ingestion container that handles pulling data from two APIs and cleaning/putting them together.
    - After the data pull is finished and the flight and weather data are merged together, a POST request is made to the FastAPI server to enable pushing the output of the ingestion operation to the server.
- A preprocessing container that handles the necessary data pre-processing tasks.
    - This container makes a GET request to the FastAPI server, receiving the ingestion output.
    - It performs the necessary pre-processing tasks, and pushes the output back to the server with a POST request.
- Two modeling containers.

How does it work?

- All of these docker containers are on a shared network to handle inter-container communication.
- All of these containers are also linked to a shared volume which enables the outputs to be written to a local folder for reasons of replicating data for redundancy/replication sake.
- Apart from the server container, there is a yaml file utilizing docker compose to orchestrate the sequential trigger which runs all the containers in order.
