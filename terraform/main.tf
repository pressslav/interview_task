terraform {
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0.1"
    }
  }
}

provider "docker" {}

# --- Network Definition ---
resource "docker_network" "mlops_network" {
  name = "mlops-network"
}

# --- Image Definitions ---
resource "docker_image" "registry" {
  name = "registry:2"
}

resource "docker_image" "mlflow" {
  name = "ghcr.io/mlflow/mlflow:v2.14.1"
}

# --- Container Definitions ---

# Docker Registry Container
resource "docker_container" "registry" {
  name  = "local-docker-registry"
  image = docker_image.registry.image_id
  ports {
    internal = 5000
    external = 5002 
  }
  networks_advanced {
    name = docker_network.mlops_network.name
  }
  restart = "always"
}

# MLflow Server Container
resource "docker_container" "mlflow_server" {
  name  = "mlflow-server"
  image = docker_image.mlflow.image_id
  
  ports {
    internal = 5000 
    external = 5001 
  }
  
  networks_advanced {
    name = docker_network.mlops_network.name
  }

  command = [
    "mlflow",
    "server",
    "--host", "0.0.0.0", 
    "--port", "5000",
    "--backend-store-uri", "sqlite:////mlflow/mlflow.db"
  ]
  
  volumes {
    host_path      = "${path.cwd}/mlflow-data"
    container_path = "/mlflow"
  }
  
  restart = "always"
  depends_on = [docker_container.registry]
}
